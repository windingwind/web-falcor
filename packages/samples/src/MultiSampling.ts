/**
 * Mirrors Samples/MultiSampling: a 16-triangle disk rasterized into a multisampled target,
 * resolved and blitted on even frames, blitted straight from the multisampled texture on
 * odd ones. §9: WebGPU multisamples at 4 samples (native: 8) and not in RGBA32Float, so the
 * target is RGBA16Float.
 */

import {
    MemoryType,
    RasterPass,
    ResourceBindFlags,
    ResourceFormat,
    SampleApp,
    Topology,
    Vao,
    VertexBufferLayout,
    VertexLayout,
    Fbo,
    type RenderContext,
    type SampleAppConfig,
    type Texture,
} from "@web-falcor/falcor";

const kTriangleCount = 16;
export const kSampleCount = 4;

export class MultiSampling extends SampleApp {
    static readonly config: SampleAppConfig = { windowDesc: { width: 1024, height: 1024, resizableWindow: true, enableVSync: true, title: "Falcor multi-sampling example" } };
    private rasterPass: RasterPass | null = null;
    private vao: Vao | null = null;
    private fbo: Fbo | null = null;
    private resolvedTexture: Texture | null = null;
    private frame = 0;

    override onLoad(): void {
        const device = this.getDevice();
        this.rasterPass = RasterPass.create(device, { path: "Samples/MultiSampling/MultiSampling.3d.slang", vsEntry: "vsMain", psEntry: "psMain" });

        // Disk triangles.
        const vertices = new Float32Array(kTriangleCount * 3 * 2);
        for (let i = 0; i < kTriangleCount; i++) {
            const theta0 = (i / kTriangleCount) * 2 * Math.PI;
            const theta1 = ((i + 1) / kTriangleCount) * 2 * Math.PI;
            vertices.set([0, 0, Math.cos(theta0) * 0.75, Math.sin(theta0) * 0.75, Math.cos(theta1) * 0.75, Math.sin(theta1) * 0.75], i * 6);
        }
        const vertexBuffer = device.createBuffer(vertices.byteLength, ResourceBindFlags.ShaderResource | ResourceBindFlags.Vertex, MemoryType.DeviceLocal, vertices);
        const bufferLayout = new VertexBufferLayout().addElement("POSITION", 0, ResourceFormat.RG32Float, 1, 0);
        const layout = new VertexLayout().addBufferLayout(0, bufferLayout);
        this.vao = new Vao(Topology.TriangleList, layout, [vertexBuffer]);

        this.fbo = new Fbo();
        const tex = device.createTexture2DMS(128, 128, ResourceFormat.RGBA16Float, kSampleCount, 1, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        this.fbo.attachColorTarget(tex, 0);
        this.resolvedTexture = device.createTexture2D(128, 128, ResourceFormat.RGBA16Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    }

    override onFrameRender(renderContext: RenderContext, targetFbo: Fbo): void {
        renderContext.clearFbo(this.fbo!, [0, 0, 0, 0], 0, 0);
        this.rasterPass!.getState().setVao(this.vao!);
        this.rasterPass!.draw(renderContext, this.fbo!, kTriangleCount * 3);
        if (this.frame++ % 2 === 0) {
            // For even frames, resolve to texture and then blit.
            renderContext.resolveResource(this.fbo!.getColorTexture(0)!, this.resolvedTexture!);
            renderContext.blit(this.resolvedTexture!, targetFbo.getColorTexture(0)!);
        } else {
            // For odd frames, blit directly.
            renderContext.blit(this.fbo!.getColorTexture(0)!, targetFbo.getColorTexture(0)!);
        }
    }
}
