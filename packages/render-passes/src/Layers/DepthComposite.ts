/**
 * DepthComposite: composites a layer that carries its own depth (raymarched/SDF objects, volumes, clouds, particles,
 * overlays) into the scene colour by a per-pixel depth test.
 *
 * Depth convention: smaller = nearer, and the layer depth must use the same convention as the scene depth. Scene depth
 * comes from `posW` when connected (linear view depth: distance along the camera's forward axis; background = +inf),
 * else from `depth` (channel r, e.g. a device depth buffer), else +inf. Without `layerDepth` the layer is always in
 * front (a 2D overlay).
 *
 * Blending where the layer is visible (k = visibility x opacity):
 *   premultiplied (default): out = src * lerp(1, 1 - a, k) + layer.rgb * k
 *   straight               : out = src * lerp(1, 1 - a, k) + layer.rgb * a * k
 *   additive               : out = src + layer.rgb * k
 * A connected `transmittance` (rgb) replaces the (1 - a) term (volumes: out = src * T + inscatter). visibility is 1/0
 * from the depth test, or with softness > 0 a linear fade over `softness` depth units in front of the scene surface
 * (soft particles). Output alpha = src alpha. `dstDepth` (r) = the nearer of scene and layer depth where the layer has
 * alpha > 0 and is visible, else the scene depth.
 *
 * Identity: opacity 0, or a layer with rgb = 0 and alpha = 0, reproduces src bit for bit.
 *
 * Inputs : src, layer (rgba), depth? posW? (scene), layerDepth?, transmittance?
 * Outputs: dst (RGBA32Float), dstDepth (R in RGBA32Float, optional)
 * Params : alphaMode {premultiplied|straight|additive}, opacity [0, 1], softness (>= 0, depth units)
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, cameraData, dummyTexture, enumIndex } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Layers/DepthComposite.cs.slang";
const kAlphaModes = ["premultiplied", "straight", "additive"] as const;

export class DepthComposite extends RenderPass {
    private p = { alphaMode: "premultiplied" as string, opacity: 1, softness: 0 };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof typeof this.p)[]) (this.p as Record<string, unknown>)[k] = props.get(k, this.p[k]);
        enumIndex("DepthComposite", "alphaMode", this.p.alphaMode, kAlphaModes);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Opacity", this.p.opacity, 0, 1, 0.01, (v) => (this.p.opacity = v));
        ui.slider("Softness", this.p.softness, 0, 10, 0.01, (v) => (this.p.softness = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "src", "Scene colour");
        addTexInput(r, "layer", "Layer colour (rgba)");
        addTexInput(r, "depth", "Scene depth (r; used when posW is not connected)", true);
        addTexInput(r, "posW", "Scene world position (scene depth = linear view depth)", true);
        addTexInput(r, "layerDepth", "Layer depth (r), same convention as the scene depth", true);
        addTexInput(r, "transmittance", "Layer transmittance (rgb), replaces 1 - alpha", true);
        addColorOutput(r, "dst", "Composited colour", w, h);
        addColorOutput(r, "dstDepth", "Composited depth (r)", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), layer = rd.getTexture("layer"), dst = rd.getTexture("dst");
        if (!src || !layer || !dst) return;
        const depth = rd.getTexture("depth"), posW = rd.getTexture("posW"), layerDepth = rd.getTexture("layerDepth");
        const trans = rd.getTexture("transmittance"), dstDepth = rd.getTexture("dstDepth");
        if (posW && !this.scene) throw new RuntimeError("DepthComposite: posW needs a scene (camera)");
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gAlphaMode"] = enumIndex("DepthComposite", "alphaMode", this.p.alphaMode, kAlphaModes);
        cb["gOpacity"] = Math.min(Math.max(Number(this.p.opacity), 0), 1);
        cb["gSoftness"] = Math.max(Number(this.p.softness), 0);
        cb["gSceneDepthFromPosW"] = posW ? 1 : 0;
        cb["gHasSceneDepth"] = depth ? 1 : 0;
        cb["gHasLayerDepth"] = layerDepth ? 1 : 0;
        cb["gHasTransmittance"] = trans ? 1 : 0;
        const cam = this.scene ? cameraData(this.scene) : { pos: [0, 0, 0], fwd: [0, 0, -1] };
        cb["gCamPos"] = cam.pos;
        cb["gCamFwd"] = cam.fwd;
        const dummy = dummyTexture(this.device);
        root["gSrc"] = src;
        root["gLayer"] = layer;
        root["gSceneDepth"] = depth ?? dummy;
        root["gScenePosW"] = posW ?? dummy;
        root["gLayerDepth"] = layerDepth ?? dummy;
        root["gTransmittance"] = trans ?? dummy;
        root["gDst"] = dst;
        root["gDstDepth"] = dstDepth ?? dummyTexture(this.device, "out:dstDepth");
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("DepthComposite", (device, props) => new DepthComposite(device, props), {
    desc: "Composites a colour+depth layer into the scene by depth test (premultiplied/straight/additive, transmittance)",
});
