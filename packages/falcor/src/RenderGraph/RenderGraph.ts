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
import { Buffer } from "../Core/API/Buffer.js";
import type { Resource } from "../Core/API/Resource.js";
import { MemoryType, ResourceBindFlags } from "../Core/API/Types.js";
import { ResourceFormat } from "../Core/API/Formats.js";
import { RenderPass, RenderData, createPass, type CompileData } from "./RenderPass.js";
import type { Properties } from "../Utils/Properties.js";
import { Field, FieldType, RenderPassReflection, resourceTypeToFieldType } from "./RenderPassReflection.js";
import { ArgumentError, RuntimeError } from "../Core/Error.js";
import { Logger } from "../Utils/Logger.js";

/** One graph edge, srcPass.srcField -> dstPass.dstField (native: RenderGraph::EdgeData). */
export interface RenderGraphEdge {
    srcPass: string;
    srcField: string;
    dstPass: string;
    dstField: string;
}

interface CompiledPass {
    name: string;
    pass: RenderPass;
    /** field name -> allocated or externally-bound resource */
    resources: Map<string, Resource>;
}

/** TextureChannelFlags values used by markOutput (RGB is native's default). */
const kChannelsRGB = 7;
const kChannelNames: Record<number, string> = {
    1: "TextureChannelFlags.Red",
    2: "TextureChannelFlags.Green",
    4: "TextureChannelFlags.Blue",
    8: "TextureChannelFlags.Alpha",
    15: "TextureChannelFlags.RGBA",
};

export class RenderGraph {
    private passes = new Map<string, RenderPass>();
    private edges: RenderGraphEdge[] = [];
    private outputs: { pass: string; field: string; masks: Set<number> }[] = [];
    private externalInputs = new Map<string, Resource>();
    private renderSettingsKey: string | null = null;
    private compiled: CompiledPass[] | null = null;
    private allocated = new Map<string, Resource>(); // "pass.field" -> resource
    /** Persistent fields keep their resource across recompiles while the field is unchanged. */
    private persistent = new Map<string, { field: Field; resolved: string; resolve: boolean; resource: Resource }>();
    private defaultDims: [number, number] = [1920, 1080];
    /** Format for Unknown-format outputs (native: swapchain format; web keeps float for oracle parity). */
    private defaultFormat = ResourceFormat.RGBA32Float;

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

    /** Mirrors RenderGraph::updatePass: recreates the pass from `props` alone (unset properties revert to defaults). */
    updatePass(name: string, props: Properties | Record<string, unknown>): void {
        const old = this.passes.get(name);
        if (!old) throw new ArgumentError(`Can't update render pass '${name}'. Pass doesn't exist.`);
        const pass = createPass(this.device, old.type || old.constructor.name, props);
        pass.name = name;
        this.passes.set(name, pass);
        if (this.scene) pass.setScene(this.scene);
        this.compiled = null;
    }

    removePass(name: string): void {
        this.passes.delete(name);
        this.edges = this.edges.filter((e) => e.srcPass !== name && e.dstPass !== name);
        this.compiled = null;
    }

    getPass(name: string): RenderPass | undefined {
        return this.passes.get(name);
    }

    /** Pass entries (name→pass) in insertion order, for UI/introspection. */
    getPasses(): { name: string; pass: RenderPass }[] {
        return [...this.passes].map(([name, pass]) => ({ name, pass }));
    }

    /** Edges in addEdge order (fresh copies), for UI/introspection. */
    getEdges(): RenderGraphEdge[] {
        return this.edges.map((e) => ({ ...e }));
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

    /** Mirrors RenderGraph::removeEdge(src, dst). */
    removeEdge(src: string, dst: string): void {
        const [srcPass, srcField] = splitFieldRef(src);
        const [dstPass, dstField] = splitFieldRef(dst);
        this.edges = this.edges.filter(
            (e) => !(e.srcPass === srcPass && e.srcField === srcField && e.dstPass === dstPass && e.dstField === dstField),
        );
        this.compiled = null;
    }

    /**
     * Mirrors RenderGraph::markOutput: `mask` (TextureChannelFlags, default RGB) selects
     * what frame capture writes; marking again adds a mask; "*" marks every available output.
     */
    markOutput(ref: string, mask = kChannelsRGB): void {
        if (mask === 0) throw new Error("Mask must be non-empty");
        if (ref === "*") {
            for (const o of this.getAvailableOutputs()) this.markOutput(o, mask);
            return;
        }
        const [pass, field] = splitFieldRef(ref);
        const existing = this.outputs.find((o) => o.pass === pass && o.field === field);
        if (existing) {
            existing.masks.add(mask); // already generated: no recompile
            return;
        }
        this.outputs.push({ pass, field, masks: new Set([mask]) });
        this.compiled = null;
    }

    /** Mirrors RenderGraph::getOutputMasks (by output index, in mark order). */
    getOutputMasks(index: number): Set<number> {
        return new Set(this.outputs[index]?.masks ?? []);
    }

    /** Mirrors RenderGraph::getAvailableOutputs: every output field of every pass, "Pass.field". */
    getAvailableOutputs(): string[] {
        const out: string[] = [];
        for (const [name, pass] of this.passes) {
            const compileData: CompileData = { defaultTexDims: this.defaultDims, defaultTexFormat: this.defaultFormat, connectedResources: new RenderPassReflection() };
            for (const f of pass.reflect(compileData).fields) if (f.isOutput()) out.push(`${name}.${f.name_}`);
        }
        return out;
    }

    /** Mirrors RenderGraph::unmarkOutput. */
    unmarkOutput(ref: string): void {
        const [pass, field] = splitFieldRef(ref);
        this.outputs = this.outputs.filter((o) => !(o.pass === pass && o.field === field));
        this.compiled = null;
    }

    /**
     * Mirrors RenderGraphExporter: a python script reproducing this graph.
     * Divergence (docs §9): emits the camelCase dialect of the upstream
     * image-test graphs (what runGraphScript executes) rather than the native
     * snake_case IR.
     */
    exportScript(): string {
        let varName = this.name.replace(/\W/g, "_");
        if (/^\d/.test(varName)) varName = "_" + varName;
        const fn = `render_graph_${varName}`;
        const lines = ["from falcor import *", "", `def ${fn}():`, `    g = RenderGraph(${pyRepr(this.name)})`];
        for (const [name, pass] of this.passes) {
            const live = pass.getProperties();
            const props = [...live.entries()].length > 0 ? live : pass.creationProps;
            lines.push(`    g.addPass(createPass(${pyRepr(pass.type || pass.constructor.name)}, ${pyRepr(props?.toJSON() ?? {})}), ${pyRepr(name)})`);
        }
        for (const e of this.edges) lines.push(`    g.addEdge(${pyRepr(`${e.srcPass}.${e.srcField}`)}, ${pyRepr(`${e.dstPass}.${e.dstField}`)})`);
        for (const o of this.outputs)
            for (const mask of o.masks) lines.push(`    g.markOutput(${pyRepr(`${o.pass}.${o.field}`)}${mask === kChannelsRGB ? "" : `, ${kChannelNames[mask] ?? mask}`})`);
        lines.push("    return g", "", `${varName} = ${fn}()`, `try: m.addGraph(${varName})`, "except NameError: None", "");
        return lines.join("\n");
    }

    /** Mirrors RenderGraph::setInput: binds an external resource to an unconnected input. */
    setInput(ref: string, resource: Resource): void {
        this.externalInputs.set(ref, resource);
        this.compiled = null;
    }

    private scene: import("../Scene/Scene.js").Scene | null = null;

    /** Mirrors RenderGraph::setScene: forwards to all passes. */
    setScene(scene: import("../Scene/Scene.js").Scene | null): void {
        this.scene = scene;
        for (const pass of this.passes.values()) pass.setScene(scene);
        this.compiled = null;
    }

    /** Mirrors RenderGraph::onResize(targetFbo): size-0 fields and Unknown formats follow the target. */
    onResize(width: number, height: number, format?: ResourceFormat): void {
        this.defaultDims = [width, height];
        if (format !== undefined && format !== ResourceFormat.Unknown) this.defaultFormat = format;
        this.compiled = null;
    }

    /** Mirrors RenderGraph::getOutput (texture outputs; see getOutputResource for buffers). */
    getOutput(ref: string): Texture | undefined {
        const r = this.allocated.get(ref);
        return r instanceof Texture ? r : undefined;
    }

    getOutputResource(ref: string): Resource | undefined {
        return this.allocated.get(ref);
    }

    /** Marked output refs ("Pass.field"), in mark order (RenderGraph::getOutputName). */
    getOutputNames(): string[] {
        return this.outputs.map((o) => `${o.pass}.${o.field}`);
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
        this.allocated.clear();

        // Per-pass CompileData carrying connectedResources: the already-reflected
        // source fields feeding this pass's inputs, renamed to the input field
        // (mirrors RenderGraphCompiler::prepPassCompilationData). Topological
        // order guarantees sources reflect before consumers.
        const reflections = new Map<string, ReturnType<RenderPass["reflect"]>>();
        const compileDatas = new Map<string, CompileData>();
        for (const name of order) {
            const connected = new RenderPassReflection();
            for (const e of this.edges) {
                if (e.dstPass !== name) continue;
                const srcField = reflections.get(e.srcPass)?.getField(e.srcField);
                if (srcField) connected.addConnectedField(e.dstField, srcField);
            }
            for (const [ref, res] of this.externalInputs) {
                if (!ref.startsWith(`${name}.`)) continue;
                const f = connected.addInput(ref.slice(name.length + 1), "External input resource");
                if (res instanceof Texture) {
                    f.format(res.format).resourceType(resourceTypeToFieldType(res.type), res.width, res.height, res.depth, res.sampleCount, res.mipCount, res.arraySize);
                } else if (res instanceof Buffer) {
                    f.rawBuffer(res.size);
                }
            }
            const compileData: CompileData = { defaultTexDims: this.defaultDims, defaultTexFormat: this.defaultFormat, connectedResources: connected };
            compileDatas.set(name, compileData);
            reflections.set(name, this.passes.get(name)!.reflect(compileData));
        }

        const compiled: CompiledPass[] = [];
        const livePersistent = new Set<string>();
        for (const name of order) {
            const pass = this.passes.get(name)!;
            const reflection = reflections.get(name)!;
            const resources = new Map<string, Resource>();

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

                // Optional outputs are only allocated when consumed: connected to an
                // edge or marked as a graph output (RenderGraphCompiler::isResourceUsed —
                // unallocated optional outputs read back as null in RenderData, which
                // passes use to drive their is_valid_* defines).
                const isGraphOutput = this.outputs.some((o) => o.pass === name && o.field === field.name_);
                if (field.isOptional() && field.isOutput() && !field.isInput()) {
                    const consumed = isGraphOutput || this.edges.some((e) => e.srcPass === name && e.srcField === field.name_);
                    if (!consumed) continue;
                }

                // Output / internal / input-output: allocate, merging the connected inputs'
                // requirements (ResourceCache::registerField alias path).
                const merged = field.clone();
                if (isGraphOutput && merged.bindFlags_ !== ResourceBindFlags.None) merged.bindFlags_ |= ResourceBindFlags.ShaderResource;
                // Resolve bind flags if the output or any connected input left them None
                // (native computes this per alias before merge; merge ORs the flags).
                let resolve = field.bindFlags_ === ResourceBindFlags.None;
                for (const e of this.edges.filter((e) => e.srcPass === name && e.srcField === field.name_)) {
                    const dstField = reflections.get(e.dstPass)?.getField(e.dstField);
                    if (!dstField) continue;
                    resolve ||= dstField.bindFlags_ === ResourceBindFlags.None;
                    merged.merge(dstField);
                }
                if (!merged.isValid()) throw new RuntimeError(`RenderGraph: field '${key}' is invalid`);
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

                let resource: Resource;
                const kept = merged.isPersistent() ? this.persistent.get(key) : undefined;
                const resolved = this.resolvedKey(merged);
                if (kept && kept.field.equals(merged) && kept.resolved === resolved && kept.resolve === resolve) {
                    resource = kept.resource;
                } else {
                    resource = this.allocateResource(merged, resolve);
                    if (merged.isPersistent()) this.persistent.set(key, { field: merged, resolved, resolve, resource });
                }
                if (merged.isPersistent()) livePersistent.add(key);
                resources.set(field.name_, resource);
                this.allocated.set(key, resource);
            }

            pass.compile(ctx, compileDatas.get(name)!);
            compiled.push({ name, pass, resources });
        }
        for (const key of this.persistent.keys()) if (!livePersistent.has(key)) this.persistent.delete(key);
        this.compiled = compiled;
        Logger.info(`RenderGraph '${this.name}' compiled: ${order.join(" -> ")}`);
    }

    /** Graph-default-dependent part of a field's allocation (size-0 dims, Unknown format). */
    private resolvedKey(field: Field): string {
        const format = field.format_ === ResourceFormat.Unknown ? this.defaultFormat : field.format_;
        return `${field.width || this.defaultDims[0]}x${field.height || this.defaultDims[1]}|${format}`;
    }

    /** Mirrors ResourceCache::createResourceForPass (resolveBindFlags comes from registerField, see compile). */
    private allocateResource(field: Field, resolveBindFlags: boolean): Resource {
        let bindFlags = field.bindFlags_;

        if (field.type_ === FieldType.RawBuffer) {
            if (resolveBindFlags) bindFlags = ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource;
            return new Buffer(this.device, { size: field.width, bindFlags, memoryType: MemoryType.DeviceLocal, name: field.name_ });
        }

        let format = field.format_ === ResourceFormat.Unknown ? this.defaultFormat : field.format_;
        // WebGPU has no r8uint/r16uint storage textures: promote UAV-bound
        // narrow uint formats to r32uint (uint reads are value-identical).
        const wantsUav = resolveBindFlags || (bindFlags & ResourceBindFlags.UnorderedAccess) !== 0;
        if (wantsUav && (format === ResourceFormat.R8Uint || format === ResourceFormat.R16Uint)) {
            format = ResourceFormat.R32Uint;
        }
        if (resolveBindFlags) {
            let mask = ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource;
            // WebGPU cannot render to 1D textures.
            if ((field.isOutput() || field.isInternal()) && field.type_ !== FieldType.Texture1D) mask |= ResourceBindFlags.DepthStencil | ResourceBindFlags.RenderTarget;
            bindFlags |= mask & this.device.getFormatBindFlags(format);
        }
        return new Texture(this.device, {
            type: field.getResourceType(),
            width: field.width || this.defaultDims[0],
            height: field.height || this.defaultDims[1],
            depth: field.depth || 1,
            arraySize: field.arraySize || 1,
            mipLevels: field.mipCount || 1,
            sampleCount: field.sampleCount || 1,
            format,
            bindFlags,
            name: field.name_,
        });
    }

    /**
     * Awaits async pass initialization (asset loading). Call once before the
     * first execute (web divergence, docs §9).
     */
    async init(): Promise<void> {
        await Promise.all([...this.passes.values()].map((p) => p.initAsync()));
        this.compiled = null; // reflection may depend on loaded assets (e.g. ImageLoader size)
    }

    /** Graph-wide pass dictionary (mirrors InternalDictionary; persists across frames). */
    private readonly passDictionary = new Map<string, unknown>();

    /** Mirrors RenderGraphExe::execute (plus Mogwai's per-frame scene tick). */
    execute(ctx: RenderContext): void {
        // Mirrors RenderGraph::execute's mRecompile check (passes call requestRecompile).
        for (const pass of this.passes.values()) {
            if (pass.recompileRequested) {
                pass.recompileRequested = false;
                this.compiled = null;
            }
        }
        // Mirrors IScene::UpdateFlags::RenderSettingsChanged -> passes recreate their
        // programs (scene light-usage defines changed); setScene drops cached kernels.
        if (this.scene) {
            const key = this.scene.getRenderSettingsKey();
            if (this.renderSettingsKey !== null && key !== this.renderSettingsKey) {
                for (const pass of this.passes.values()) pass.setScene(this.scene);
            }
            this.renderSettingsKey = key;
        }
        if (!this.compiled) this.compile(ctx);
        // Native Mogwai calls Scene::update() (camera beginFrame: jitter pattern
        // advance) before executing the graph each frame; web folds it in here.
        this.scene?.camera.beginFrame();
        const profiler = ctx.device.profilerHook;
        profiler?.startEvent("RenderGraphExe::execute()");
        for (const { pass, resources } of this.compiled!) {
            const label = pass.name || pass.constructor.name;
            profiler?.startEvent(label);
            pass.execute(ctx, new RenderData(resources, this.defaultDims, this.passDictionary));
            profiler?.endEvent(label);
        }
        profiler?.endEvent("RenderGraphExe::execute()");
        // Web: the graph closes the profiler frame (native SampleApp does it once per frame).
        profiler?.endFrame(ctx.getEncoder());
    }
}

/** Python literal for a property value (vectors as floatN(...) factory calls). */
function pyRepr(v: unknown): string {
    if (typeof v === "boolean") return v ? "True" : "False";
    if (typeof v === "number") return Number.isFinite(v) ? String(v) : `float("${v > 0 ? "inf" : v < 0 ? "-inf" : "nan"}")`;
    if (typeof v === "string") return JSON.stringify(v);
    if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
    if (v && typeof v === "object") {
        const o = v as Record<string, unknown>;
        const comps = ["x", "y", "z", "w"].filter((c) => typeof o[c] === "number");
        if (comps.length >= 2 && Object.keys(o).length === comps.length) {
            return `float${comps.length}(${comps.map((c) => o[c]).join(", ")})`;
        }
        return `{${Object.entries(o)
            .map(([k, val]) => `${JSON.stringify(k)}: ${pyRepr(val)}`)
            .join(", ")}}`;
    }
    return "None";
}

function splitFieldRef(ref: string): [string, string] {
    const idx = ref.lastIndexOf(".");
    if (idx <= 0) throw new ArgumentError(`Invalid field reference '${ref}' (expected "pass.field")`);
    return [ref.slice(0, idx), ref.slice(idx + 1)];
}
