/**
 * MotionBlur: post-process motion blur from screen-space motion vectors with the reconstruction filter of
 * McGuire, Hennessy, Bukowski and Osman (2012, "A Reconstruction Filter for Plausible Motion Blur"):
 * tile-max velocity (tileSize x tileSize), 3x3 neighbour max, then a jittered gather of sampleCount taps along the
 * neighbourhood's dominant velocity with soft depth comparison (foreground/background) and cone/cylinder weights.
 * Native Falcor 8 has no motion-blur pass; this is a web-falcor addition.
 *
 * Inputs : color (linear colour), mvec (GBufferRaster/GBufferRT mvec: previous minus current position in UV units),
 *          posW (GBuffer posW, w = 0: no geometry; view depth = (posW - camera position) . camera forward).
 * Outputs: dst (RGBA32Float).
 * The blur spans the motion during an exposure of `shutter` frame intervals centred on the current frame
 * (half-extent V = 0.5 * shutter * mvec * frameDim pixels, clamped to tileSize). Pixels without geometry carry the
 * G-buffer's cleared mvec (0), so the sky itself does not streak under camera rotation.
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { cameraFrame } from "../DepthOfField/DepthOfField.js";

const kShaderFile = "RenderPasses/MotionBlur/MotionBlur.cs.slang";

export class MotionBlur extends RenderPass {
    private p = { shutter: 0.5, tileSize: 20, sampleCount: 15, softZ: 0.02, noise: true };
    private tilePass: ComputePass | null = null;
    private neighborPass: ComputePass | null = null;
    private gatherPass: ComputePass | null = null;
    private tileMax: Texture | null = null;
    private neighborMax: Texture | null = null;
    private frame = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof typeof this.p)[]) (this.p as Record<string, unknown>)[k] = props.get(k, this.p[k]);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Shutter (frames)", this.p.shutter, 0, 2, 0.01, (v) => (this.p.shutter = v));
        ui.slider("Tile size / max blur (px)", this.p.tileSize, 2, 64, 1, (v) => (this.p.tileSize = v));
        ui.slider("Samples", this.p.sampleCount, 3, 63, 2, (v) => (this.p.sampleCount = v));
        ui.slider("Soft depth (relative)", this.p.softZ, 0.001, 0.5, 0.001, (v) => (this.p.softZ = v));
        ui.checkbox("Noise", this.p.noise, (v) => (this.p.noise = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("color", "Linear colour").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("mvec", "Screen-space motion vectors (UV units, previous - current)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("posW", "World position (w = 0: background)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Motion-blurred colour")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    private tex(old: Texture | null, w: number, h: number, name: string): Texture {
        if (old && old.width === w && old.height === h) return old;
        return new Texture(this.device, {
            type: ResourceType.Texture2D, width: w, height: h, format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource, name,
        });
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color");
        const mvec = rd.getTexture("mvec");
        const posW = rd.getTexture("posW");
        const dst = rd.getTexture("dst");
        if (!color || !mvec || !posW || !dst) return;
        const W = dst.width, H = dst.height;
        const K = Math.max(2, Math.round(this.p.tileSize));
        const TW = Math.ceil(W / K), TH = Math.ceil(H / K);
        this.tileMax = this.tex(this.tileMax, TW, TH, "MotionBlur::tileMax");
        this.neighborMax = this.tex(this.neighborMax, TW, TH, "MotionBlur::neighborMax");
        if (!this.tilePass) {
            this.tilePass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "tileMaxMain" });
            this.neighborPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "neighborMaxMain" });
            this.gatherPass = ComputePass.create(this.device, { path: kShaderFile, csEntry: "gatherMain" });
        }
        const [camPos, camFwd] = cameraFrame(this.scene);
        const passes = [this.tilePass, this.neighborPass!, this.gatherPass!];
        for (const pass of passes) {
            const cb = pass.getRootVar()["CB"]!;
            cb["gFrameDim"] = [W, H];
            cb["gTileDim"] = [TW, TH];
            cb["gTileSize"] = K;
            cb["gHalfExtentScale"] = 0.5 * this.p.shutter;
            cb["gSampleCount"] = Math.max(3, Math.round(this.p.sampleCount) | 1);
            cb["gSoftZ"] = this.p.softZ;
            cb["gNoise"] = this.p.noise ? 1 : 0;
            cb["gFrame"] = this.frame;
            cb["gCamPos"] = camPos;
            cb["gCamFwd"] = camFwd;
        }
        // Every kernel sees every resource; slots a kernel doesn't use get a read-only stand-in, never its own output.
        const bind = (pass: ComputePass, tileIn: Texture, out: Texture) => {
            const root = pass.getRootVar();
            root["gColor"] = color;
            root["gMvec"] = mvec;
            root["gPosW"] = posW;
            root["gTileIn"] = tileIn;
            root["gOut"] = out;
        };
        bind(this.tilePass, color, this.tileMax);
        this.tilePass.execute(ctx, TW, TH);
        bind(this.neighborPass!, this.tileMax, this.neighborMax);
        this.neighborPass!.execute(ctx, TW, TH);
        bind(this.gatherPass!, this.neighborMax, dst);
        this.gatherPass!.execute(ctx, W, H);
        this.frame++;
    }
}

registerRenderPass("MotionBlur", (device, props) => new MotionBlur(device, props), { desc: "Motion blur from motion vectors (McGuire 2012 reconstruction)" });
