/**
 * Render pass base + plugin registry mirroring Falcor/RenderGraph/RenderPass.h
 * and the Plugin system (ES-module self-registration replaces DLL plugins).
 */

import type { Device } from "../Core/API/Device.js";
import type { RenderContext } from "../Core/API/RenderContext.js";
import type { Texture } from "../Core/API/Texture.js";
import { Properties } from "../Utils/Properties.js";
import { RenderPassReflection } from "./RenderPassReflection.js";
import type { Scene } from "../Scene/Scene.js";
import { RuntimeError } from "../Core/Error.js";

export interface CompileData {
    /** Default output dimensions (mirrors RenderPassHelpers::IOSize default). */
    defaultTexDims: [number, number];
}

/** Resource dictionary passed to execute (mirrors Falcor::RenderData). */
export class RenderData {
    constructor(
        private readonly resources: Map<string, Texture>,
        public readonly defaultTexDims: [number, number],
    ) {}

    getTexture(name: string): Texture | undefined {
        return this.resources.get(name);
    }
}

export abstract class RenderPass {
    name = "";

    constructor(public readonly device: Device) {}

    /** Mirrors RenderPass::reflect. */
    abstract reflect(compileData: CompileData): RenderPassReflection;

    /** Mirrors RenderPass::execute. */
    abstract execute(ctx: RenderContext, renderData: RenderData): void;

    /** Mirrors RenderPass::compile (called when graph recompiles). */
    compile(_ctx: RenderContext, _compileData: CompileData): void {}

    /**
     * Async initialization (asset loading etc.) — web divergence (DESIGN.md §9):
     * native Falcor blocks on file IO in constructors. Awaited by RenderGraph.init().
     */
    async initAsync(): Promise<void> {}

    protected scene: Scene | null = null;

    /** Mirrors RenderPass::setScene. */
    setScene(scene: Scene | null): void {
        this.scene = scene;
    }

    setProperties(_props: Properties): void {}
    getProperties(): Properties {
        return new Properties();
    }
}

export type RenderPassFactory = (device: Device, props: Properties) => RenderPass;

const registry = new Map<string, RenderPassFactory>();

/** Mirrors the plugin registration done in each pass's registerPlugin(). */
export function registerRenderPass(type: string, factory: RenderPassFactory): void {
    registry.set(type, factory);
}

/** Mirrors createPass() from the scripting API. */
export function createPass(device: Device, type: string, props: Properties | Record<string, unknown> = {}): RenderPass {
    const factory = registry.get(type);
    if (!factory) throw new RuntimeError(`Unknown render pass type '${type}'. Registered: ${[...registry.keys()].join(", ")}`);
    const properties = props instanceof Properties ? props : new Properties(props as Record<string, never>);
    const pass = factory(device, properties);
    pass.name = type;
    return pass;
}

export function getRegisteredRenderPasses(): string[] {
    return [...registry.keys()];
}
