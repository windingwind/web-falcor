/**
 * DepthOfField: gather-based thin-lens depth of field in one compute dispatch ("scatter as gather" over a golden-angle
 * spiral, after D. Gustafsson 2018, "Bokeh depth of field in a single pass").
 * Native Falcor 8 only has ray-traced DoF (thin-lens camera rays in GBufferRT / the path tracers); this raster
 * counterpart uses the same camera model, so it can be checked against GBufferRT with apertureRadius > 0.
 *
 * Inputs : color (linear colour), depth (GBuffer linearZ: x = view-space depth along the camera axis; 0 = no geometry).
 * Outputs: dst (RGBA32Float).
 * Circle of confusion, in pixels, for Falcor's thin-lens camera (lens radius A, focal plane at distance zf):
 *     coc(z) = A * |z - zf| / (z * zf) * H / (2 tan(fovY / 2)),   clamped to maxCoC.
 * With useCamera (default) A and zf are the scene camera's apertureRadius and focalDistance; otherwise the
 * apertureRadius / focalDistance properties. Background pixels (depth 0) use farDepth.
 * Each pixel sums taps on a spiral out to maxCoC; a tap counts if its own CoC reaches the pixel (smoothstep over
 * one pixel), and taps behind the pixel have their CoC clamped to twice the centre's (limits background bleeding
 * into sharp foreground). Sample density: the spiral radius grows by radiusScale / r per tap
 * (about (maxCoC / radiusScale)^2 / 2 taps).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/DepthOfField/DepthOfField.cs.slang";

export class DepthOfField extends RenderPass {
    private p = { useCamera: true, apertureRadius: 0.05, focalDistance: 5.0, maxCoC: 24, radiusScale: 1.0, farDepth: 1e6, fovY: 45 };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof typeof this.p)[]) (this.p as Record<string, unknown>)[k] = props.get(k, this.p[k]);
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.checkbox("Use camera lens", this.p.useCamera, (v) => (this.p.useCamera = v));
        ui.slider("Aperture radius", this.p.apertureRadius, 0, 1, 0.001, (v) => (this.p.apertureRadius = v));
        ui.slider("Focal distance", this.p.focalDistance, 0.01, 1000, 0.01, (v) => (this.p.focalDistance = v));
        ui.slider("Max CoC (px)", this.p.maxCoC, 1, 64, 1, (v) => (this.p.maxCoC = v));
        ui.slider("Radius scale", this.p.radiusScale, 0.25, 4, 0.01, (v) => (this.p.radiusScale = v));
    }

    /** Lens parameters in effect: [apertureRadius, focalDistance, fovY (rad)]. */
    lens(): [number, number, number] {
        const cam = this.scene?.camera;
        const fovY = cam ? cam.getFovY() : (this.p.fovY * Math.PI) / 180;
        if (this.p.useCamera && cam) return [cam.getApertureRadius(), cam.getFocalDistance(), fovY];
        return [this.p.apertureRadius, this.p.focalDistance, fovY];
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("color", "Linear colour").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("depth", "Linear view depth (GBuffer linearZ)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Colour with depth of field")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color");
        const depth = rd.getTexture("depth");
        const dst = rd.getTexture("dst");
        if (!color || !depth || !dst) return;
        if (!this.pass) this.pass = ComputePass.create(this.device, { path: kShaderFile });
        const [A, zf, fovY] = this.lens();
        const W = dst.width, H = dst.height;
        const cb = this.pass.getRootVar()["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gCocScale"] = (A * H) / (2 * Math.tan(0.5 * fovY) * Math.max(zf, 1e-6)); // coc = scale * |z - zf| / z
        cb["gFocalDistance"] = zf;
        cb["gMaxCoC"] = this.p.maxCoC;
        cb["gRadiusScale"] = Math.max(0.05, this.p.radiusScale);
        cb["gFarDepth"] = this.p.farDepth;
        const root = this.pass.getRootVar();
        root["gColor"] = color;
        root["gDepth"] = depth;
        root["gDst"] = dst;
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("DepthOfField", (device, props) => new DepthOfField(device, props), { desc: "Gather-based thin-lens depth of field" });
