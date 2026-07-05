/**
 * Render graph mirroring Falcor/RenderGraph/RenderGraph.h (+ the essential
 * parts of RenderGraphCompiler and ResourceCache).
 *
 * Divergence note: transient resources are allocated per-field without memory
 * aliasing (Falcor's ResourceCache lifetime analysis); correctness-identical,
 * more memory. Tracked for optimization later.
 */

import type { Device } from "../Core/API/Device.js";
import type { RenderContext } from "../Core/API/RenderContext.js";
import { Texture } from "../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType } from "../Core/API/Types.js";
import { ResourceFormat } from "../Core/API/Formats.js";
import { RenderPass, RenderData, type CompileData } from "./RenderPass.js";
import { Field, FieldVisibility } from "./RenderPassReflection.js";
import { ArgumentError, RuntimeError } from "../Core/Error.js";
import { Logger } from "../Utils/Logger.js";

interface Edge {
    srcPass: string;
    srcField: string;
    dstPass: string;
    dstField: string;
}

interface CompiledPass {
    name: string;
    pass: RenderPass;
    /** field name -> allocated or externally-bound texture */
    resources: Map<string, Texture>;
}

export class RenderGraph {
    private passes = new Map<string, RenderPass>();
    private edges: Edge[] = [];
    private outputs: { pass: string; field: string }[] = [];
    private externalInputs = new Map<string, Texture>();
    private compiled: CompiledPass[] | null = null;
    private allocated = new Map<string, Texture>(); // "pass.field" -> texture
    private defaultDims: [number, number] = [1920, 1080];

    constructor(
        public readonly device: Device,
        public readonly name = "RenderGraph",
    ) {}

    /** Mirrors RenderGraph::addPass. */
    addPass(pass: RenderPass, name: string): RenderPass {
        if (this.passes.has(name)) throw new ArgumentError(`Pass '${name}' already exists`);
        this.passes.set(name, pass);
        pass.name = name;
        if (this.scene) pass.setScene(this.scene);
        this.compiled = null;
        return pass;
    }

    removePass(name: string): void {
        this.passes.delete(name);
        this.edges = this.edges.filter((e) => e.srcPass !== name && e.dstPass !== name);
        this.compiled = null;
    }

    getPass(name: string): RenderPass | undefined {
        return this.passes.get(name);
    }

    /** Mirrors RenderGraph::addEdge("srcPass.field", "dstPass.field"). */
    addEdge(src: string, dst: string): void {
        const [srcPass, srcField] = splitFieldRef(src);
        const [dstPass, dstField] = splitFieldRef(dst);
        if (!this.passes.has(srcPass)) throw new ArgumentError(`addEdge: unknown source pass '${srcPass}'`);
        if (!this.passes.has(dstPass)) throw new ArgumentError(`addEdge: unknown destination pass '${dstPass}'`);
        this.edges.push({ srcPass, srcField, dstPass, dstField });
        this.compiled = null;
    }

    /** Mirrors RenderGraph::markOutput. */
    markOutput(ref: string): void {
        const [pass, field] = splitFieldRef(ref);
        this.outputs.push({ pass, field });
        this.compiled = null;
    }

    /** Mirrors RenderGraph::setInput: binds an external resource to an unconnected input. */
    setInput(ref: string, texture: Texture): void {
        this.externalInputs.set(ref, texture);
        this.compiled = null;
    }

    private scene: import("../Scene/Scene.js").Scene | null = null;

    /** Mirrors RenderGraph::setScene: forwards to all passes. */
    setScene(scene: import("../Scene/Scene.js").Scene | null): void {
        this.scene = scene;
        for (const pass of this.passes.values()) pass.setScene(scene);
        this.compiled = null;
    }

    /** Mirrors RenderGraph::onResize (graph output dimensions drive size-0 fields). */
    onResize(width: number, height: number): void {
        this.defaultDims = [width, height];
        this.compiled = null;
    }

    /** Mirrors RenderGraph::getOutput. */
    getOutput(ref: string): Texture | undefined {
        return this.allocated.get(ref);
    }

    /** Topological order over pass dependencies (RenderGraphCompiler::sortPasses). */
    private sortPasses(): string[] {
        const inDegree = new Map<string, number>();
        const adj = new Map<string, Set<string>>();
        for (const name of this.passes.keys()) {
            inDegree.set(name, 0);
            adj.set(name, new Set());
        }
        for (const e of this.edges) {
            if (!adj.get(e.srcPass)!.has(e.dstPass)) {
                adj.get(e.srcPass)!.add(e.dstPass);
                inDegree.set(e.dstPass, inDegree.get(e.dstPass)! + 1);
            }
        }
        const queue = [...this.passes.keys()].filter((n) => inDegree.get(n) === 0);
        const order: string[] = [];
        while (queue.length) {
            const n = queue.shift()!;
            order.push(n);
            for (const m of adj.get(n)!) {
                inDegree.set(m, inDegree.get(m)! - 1);
                if (inDegree.get(m) === 0) queue.push(m);
            }
        }
        if (order.length !== this.passes.size) throw new RuntimeError("Render graph contains a cycle");
        return order;
    }

    /** Mirrors RenderGraphCompiler::compile + ResourceCache::allocateResources. */
    compile(ctx: RenderContext): void {
        const order = this.sortPasses();
        const compileData: CompileData = { defaultTexDims: this.defaultDims };
        this.allocated.clear();

        const reflections = new Map<string, ReturnType<RenderPass["reflect"]>>();
        for (const name of order) reflections.set(name, this.passes.get(name)!.reflect(compileData));

        const compiled: CompiledPass[] = [];
        for (const name of order) {
            const pass = this.passes.get(name)!;
            const reflection = reflections.get(name)!;
            const resources = new Map<string, Texture>();

            for (const field of reflection.fields) {
                const key = `${name}.${field.name_}`;

                if (field.isInput() && !field.isOutput()) {
                    const edge = this.edges.find((e) => e.dstPass === name && e.dstField === field.name_);
                    if (edge) {
                        const src = this.allocated.get(`${edge.srcPass}.${edge.srcField}`);
                        if (!src) throw new RuntimeError(`Edge source ${edge.srcPass}.${edge.srcField} not allocated`);
                        resources.set(field.name_, src);
                        continue;
                    }
                    const external = this.externalInputs.get(key);
                    if (external) {
                        resources.set(field.name_, external);
                        continue;
                    }
                    if (!field.isOptional()) {
                        throw new RuntimeError(`Required input '${key}' is not connected`);
                    }
                    continue;
                }

                // Output / internal / input-output: allocate (merging connected inputs' requirements).
                const merged = new Field(field.name_, field.desc_, field.visibility_).merge(field);
                merged.resourceType = field.resourceType;
                merged.width = field.width;
                merged.height = field.height;
                merged.depth = field.depth;
                merged.mipCount = field.mipCount;
                merged.arraySize = field.arraySize;
                merged.sampleCount = field.sampleCount;
                for (const e of this.edges.filter((e) => e.srcPass === name && e.srcField === field.name_)) {
                    const dstField = reflections.get(e.dstPass)?.getField(e.dstField);
                    if (dstField) merged.merge(dstField);
                }
                // Input-output passthrough: bind the connected source instead of allocating.
                if (field.isInput() && field.isOutput()) {
                    const edge = this.edges.find((e) => e.dstPass === name && e.dstField === field.name_);
                    if (edge) {
                        const src = this.allocated.get(`${edge.srcPass}.${edge.srcField}`)!;
                        resources.set(field.name_, src);
                        this.allocated.set(key, src);
                        continue;
                    }
                }

                const texture = this.allocateField(merged);
                resources.set(field.name_, texture);
                this.allocated.set(key, texture);
            }

            pass.compile(ctx, compileData);
            compiled.push({ name, pass, resources });
        }
        this.compiled = compiled;
        Logger.info(`RenderGraph '${this.name}' compiled: ${order.join(" -> ")}`);
    }

    private allocateField(field: Field): Texture {
        const format = field.format_ === ResourceFormat.Unknown ? ResourceFormat.RGBA32Float : field.format_;
        const bindFlags =
            field.bindFlags_ === ResourceBindFlags.None
                ? ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget
                : field.bindFlags_;
        return new Texture(this.device, {
            type: field.resourceType === ResourceType.Texture3D ? ResourceType.Texture3D : ResourceType.Texture2D,
            width: field.width || this.defaultDims[0],
            height: field.height || this.defaultDims[1],
            depth: field.depth || 1,
            arraySize: field.arraySize,
            mipLevels: field.mipCount,
            sampleCount: field.sampleCount,
            format,
            bindFlags,
            name: field.name_,
        });
    }

    /**
     * Awaits async pass initialization (asset loading). Call once before the
     * first execute (web divergence, DESIGN.md §9).
     */
    async init(): Promise<void> {
        await Promise.all([...this.passes.values()].map((p) => p.initAsync()));
        this.compiled = null; // reflection may depend on loaded assets (e.g. ImageLoader size)
    }

    /** Mirrors RenderGraphExe::execute. */
    execute(ctx: RenderContext): void {
        if (!this.compiled) this.compile(ctx);
        for (const { pass, resources } of this.compiled!) {
            pass.execute(ctx, new RenderData(resources, this.defaultDims));
        }
    }
}

function splitFieldRef(ref: string): [string, string] {
    const idx = ref.lastIndexOf(".");
    if (idx <= 0) throw new ArgumentError(`Invalid field reference '${ref}' (expected "pass.field")`);
    return [ref.slice(0, idx), ref.slice(idx + 1)];
}
