/**
 * Render pass base + plugin registry mirroring Falcor/RenderGraph/RenderPass.h
 * and the Plugin system (ES-module self-registration replaces DLL plugins).
 */

import type { OverlayDrawList } from "../Utils/UI/OverlayDrawList.js";
import type { Device } from "../Core/API/Device.js";
import type { RenderContext } from "../Core/API/RenderContext.js";
import { Texture } from "../Core/API/Texture.js";
import { Buffer } from "../Core/API/Buffer.js";
import type { Resource } from "../Core/API/Resource.js";
import type { ResourceFormat } from "../Core/API/Formats.js";
import { Properties } from "../Utils/Properties.js";
import { RenderPassReflection } from "./RenderPassReflection.js";
import type { Scene } from "../Scene/Scene.js";
import type { UIWidgets } from "./UIWidgets.js";
import { RuntimeError } from "../Core/Error.js";

export interface CompileData {
    /** Default output dimensions (mirrors RenderPassHelpers::IOSize default). */
    defaultTexDims: [number, number];
    /** Format given to Unknown-format outputs (mirrors CompileData::defaultTexFormat). */
    defaultTexFormat?: ResourceFormat;
    /**
     * Reflection of the resources connected to this pass's inputs, keyed by the
     * input field name (mirrors CompileData::connectedResources). Passes like
     * GaussianBlur shape their outputs from the incoming format/dimensions.
     */
    connectedResources?: RenderPassReflection;
}

/** Resource dictionary passed to execute (mirrors Falcor::RenderData). */
export class RenderData {
    constructor(
        private readonly resources: Map<string, Resource>,
        public readonly defaultTexDims: [number, number],
        /** Graph-wide key/value store (mirrors InternalDictionary; e.g. PRNG dimension). */
        public readonly dictionary: Map<string, unknown> = new Map(),
    ) {}

    /** Mirrors RenderData::getResource (textures and raw buffers). */
    getResource(name: string): Resource | undefined {
        return this.resources.get(name);
    }

    /** Mirrors RenderData::getTexture; undefined when unbound or a buffer field. */
    getTexture(name: string): Texture | undefined {
        const r = this.resources.get(name);
        return r instanceof Texture ? r : undefined;
    }

    /** Raw-buffer fields (Field.rawBuffer). */
    getBuffer(name: string): Buffer | undefined {
        const r = this.resources.get(name);
        return r instanceof Buffer ? r : undefined;
    }
}

/** Dictionary key: PRNG dimensions consumed upstream of a tracer (mirrors kRenderPassPRNGDimension). */
export const kRenderPassPRNGDimension = "prngDimension";

export abstract class RenderPass {
    name = "";
    /** Registry type string, set by createPass (used by RenderGraph.exportScript). */
    type = "";
    /** Creation properties (exportScript fallback for passes without getProperties). */
    creationProps: Properties | null = null;

    constructor(public readonly device: Device) {}

    /** Mirrors RenderPass::reflect. */
    abstract reflect(compileData: CompileData): RenderPassReflection;

    /** Mirrors RenderPass::execute. */
    abstract execute(ctx: RenderContext, renderData: RenderData): void;

    /** Mirrors RenderPass::compile (called when graph recompiles). */
    compile(_ctx: RenderContext, _compileData: CompileData): void {}

    /** Set by requestRecompile; the owning RenderGraph consumes it at the next execute. */
    recompileRequested = false;

    /** Mirrors RenderPass::requestRecompile: the graph recompiles before its next execute. */
    requestRecompile(): void {
        this.recompileRequested = true;
    }

    /**
     * Async initialization (asset loading etc.) — web divergence (docs §9):
     * native Falcor blocks on file IO in constructors. Awaited by RenderGraph.init().
     */
    async initAsync(): Promise<void> {}

    protected scene: Scene | null = null;

    /** Mirrors RenderPass::setScene. */
    setScene(scene: Scene | null): void {
        this.scene = scene;
    }

    setProperties(_props: Properties): void {}

    /** Mirrors RenderPass::onOptionsChange (global Settings options updated). */
    onOptionsChange(_options: Record<string, unknown>): void {}

    getProperties(): Properties {
        return new Properties();
    }

    /** Adds this pass's live controls to a UI panel (mirrors RenderPass::renderUI). */
    renderUI(_ui: UIWidgets): void {}

    /** Draws over the presented frame, even with the pass UI closed (mirrors RenderPass::renderOverlayUI). */
    renderOverlayUI(_drawList: OverlayDrawList): void {}
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
    pass.type = type;
    pass.creationProps = properties;
    return pass;
}

/**
 * Mirrors loadRenderPassLibrary / PluginManager::loadPlugin for the web: dynamically
 * imports a JS module whose top level calls registerRenderPass(); returns the pass
 * types it added. Native loads .dll/.so plugins from the plugins directory instead.
 */
export async function loadRenderPassLibrary(url: string): Promise<string[]> {
    const before = new Set(registry.keys());
    // Plugins may import the package or use this hook (blob:/data: modules can't resolve bare specifiers).
    (globalThis as { webFalcorPlugins?: unknown }).webFalcorPlugins = { registerRenderPass };
    await import(/* @vite-ignore */ url);
    return [...registry.keys()].filter((type) => !before.has(type));
}

export function getRegisteredRenderPasses(): string[] {
    return [...registry.keys()];
}
