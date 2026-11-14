/**
 * SubsetRaster: re-rasterises only a chosen set of geometry instances ("redraw only these objects") into its own
 * G-buffer targets. It is GBufferRaster restricted to the draws of `instanceMask`: the same channels (posW, normW,
 * vbuffer, depth, ...), the same shader and vertex math, so wherever a subset object is visible in the full scene its
 * subset channels equal the full G-buffer's. Hidden parts of the subset (behind other objects) are drawn too, because
 * the subset is depth-tested only against itself.
 *
 * The `vbuffer` output is always produced (the other channels stay optional, as in GBufferRaster).
 * Optional `sceneVBuffer` input + `subsetMask` output (R32Float) classify every pixel against the full scene:
 *   0 = no subset geometry, 1 = subset surface visible in the scene, 2 = subset surface hidden behind other geometry.
 * The comparison uses the hit ids (instance + triangle), not depth, so it is exact.
 *
 * Params: instanceMask (geometry instance ids, as written by StyleAovs `id.g` with idSource = instance), plus every
 * GBufferRaster parameter. The pass does not install a camera sample pattern (it follows the main G-buffer's jitter).
 * Stateless. Without this pass in the graph nothing changes; an empty mask draws nothing.
 */

import {
    ComputePass,
    FieldFlags,
    Properties,
    RenderData,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";
import { GBufferRaster } from "./GBufferRaster.js";

const kShaderFile = "RenderPasses/GBuffer/SubsetRaster.cs.slang";

type Draw = { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number };

export class SubsetRaster extends GBufferRaster {
    private instanceMask: number[] = [];
    private maskPass: ComputePass | null = null;
    private maskPassKey = "";
    private drawCache: { source: readonly Draw[]; mask: number[]; draws: Draw[] } | null = null;

    constructor(device: Device, props: Properties) {
        super(device, props);
        this.setsCameraJitter = false;
        this.setInstanceMask(props.get<number[]>("instanceMask", []));
    }

    override getProperties(): Properties {
        const p = super.getProperties();
        p.set("instanceMask", [...this.instanceMask]);
        return p;
    }

    /** Sets the geometry instance ids to draw. */
    setInstanceMask(ids: ArrayLike<number>): void {
        this.instanceMask = [...new Set(Array.from(ids, (v) => Math.round(Number(v))))].filter((v) => v >= 0).sort((a, b) => a - b);
        this.drawCache = null;
    }

    getInstanceMask(): number[] {
        return [...this.instanceMask];
    }

    protected override getDraws(): readonly Draw[] {
        const all = super.getDraws();
        if (!this.drawCache || this.drawCache.source !== all || this.drawCache.mask !== this.instanceMask) {
            const ids = new Set(this.instanceMask);
            this.drawCache = { source: all, mask: this.instanceMask, draws: all.filter((_, i) => ids.has(i)) };
        }
        return this.drawCache.draws;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = super.reflect(compileData);
        // The subset ids are always produced (they define the subset mask and are the cheapest channel to consume).
        r.getField("vbuffer")?.flags(FieldFlags.None);
        const depth = r.getField("depth");
        const [w, h] = depth ? [depth.width, depth.height] : compileData.defaultTexDims;
        r.addInput("sceneVBuffer", "V-buffer of the full scene (for subsetMask)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("subsetMask", "0 = no subset, 1 = subset visible in the scene, 2 = subset hidden (needs sceneVBuffer)")
            .texture2D(w, h)
            .format(ResourceFormat.R32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource)
            .flags(FieldFlags.Optional);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        super.execute(ctx, renderData);
        const mask = renderData.getTexture("subsetMask");
        if (!mask) return;
        const vb = renderData.getTexture("vbuffer");
        const sceneVb = renderData.getTexture("sceneVBuffer");
        if (!this.scene || !vb || !sceneVb) {
            ctx.clearTexture(mask, [0, 0, 0, 0]);
            if (this.scene && !sceneVb) throw new Error("SubsetRaster: subsetMask needs the sceneVBuffer input");
            return;
        }
        if (sceneVb.width !== mask.width || sceneVb.height !== mask.height) {
            throw new Error(`SubsetRaster: sceneVBuffer ${sceneVb.width}x${sceneVb.height} != output ${mask.width}x${mask.height}`);
        }
        const defines = this.scene.getSceneDefines();
        const key = defines.key();
        if (!this.maskPass || key !== this.maskPassKey) {
            this.maskPass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.maskPassKey = key;
        }
        const root = this.maskPass.getRootVar();
        root["CB"]!["gFrameDim"] = [mask.width, mask.height];
        root["gSubsetVBuffer"] = vb;
        root["gSceneVBuffer"] = sceneVb;
        root["gSubsetMask"] = mask;
        this.maskPass.execute(ctx, mask.width, mask.height);
    }
}

registerRenderPass("SubsetRaster", (device, props) => new SubsetRaster(device, props), {
    desc: "Re-rasterises only the geometry instances in instanceMask into separate G-buffer targets (+ visible/hidden mask)",
});
