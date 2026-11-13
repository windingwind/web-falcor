/**
 * Cascaded shadow map for one directional light (web-only raster pass; Falcor 4's
 * CSM was dropped upstream). The scene is drawn depth-only from the light once
 * per cascade into an R32Float atlas (cascade i occupies [i*size, (i+1)*size) x [0, size)).
 * Cascades split the camera's view range (clipped to the scene bounds) with the
 * practical split scheme; each cascade is a texel-snapped orthographic box around
 * the bounding sphere of its frustum slice, deep enough to hold every caster in the scene.
 * The per-cascade matrices go to the graph dictionary (kCascadedShadowMapData) where
 * DeferredLighting picks them up together with the "shadowMap" output.
 */

import {
    DepthStencilState,
    DepthStencilStateDesc,
    Fbo,
    GraphicsState,
    CullMode,
    RasterizerState,
    RasterizerStateDesc,
    ParameterBlock,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    ShaderType,
    Texture,
    Vao,
    VertexBufferLayout,
    VertexLayout,
    Topology,
    InputClass,
    LightType,
    Logger,
    float4x4,
    makeRootVar,
    mergeWgslBindings,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/RasterLighting/ShadowMap.3d.slang";
export const kMaxCascades = 4;
/** Dictionary key under which the pass publishes its CascadedShadowMapData each frame. */
export const kCascadedShadowMapData = "CascadedShadowMap.data";

export interface CascadedShadowMapData {
    lightIndex: number;
    cascadeCount: number;
    size: number;
    viewProj: float4x4[];
    /** Far view depth of each cascade (along the camera forward axis). */
    splits: number[];
    /** World-space size of one shadow texel per cascade. */
    texelWorld: number[];
    /** Light-space depth range (world units) per cascade; light depth [0,1] maps onto it. */
    depthRange: number[];
}

type V3 = [number, number, number];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a: V3): V3 => mul(a, 1 / Math.hypot(a[0], a[1], a[2]));
const v3 = (v: { x: number; y: number; z: number }): V3 => [v.x, v.y, v.z];

interface CascadeVars {
    vars: ParameterBlock;
    root: ShaderVar;
}

export class CascadedShadowMap extends RenderPass {
    private lightIndex = -1;
    private cascadeCount = 3;
    private mapSize = 2048;
    private splitLambda = 0.8;
    private maxDistance = 0;
    private useAlphaTest = true;
    private state: GraphicsState | null = null;
    private pipelineLayout: GPUPipelineLayout | null = null;
    private cascades: CascadeVars[] = [];
    private vao: Vao | null = null;
    private depth: Texture | null = null;
    private warned = false;

    constructor(device: Device, props: Properties) {
        super(device);
        this.lightIndex = props.get("lightIndex", -1);
        this.cascadeCount = Math.min(kMaxCascades, Math.max(1, props.get("cascadeCount", 3)));
        this.mapSize = props.get("mapSize", 2048);
        this.splitLambda = props.get("splitLambda", 0.8);
        this.maxDistance = props.get("maxDistance", 0);
        this.useAlphaTest = props.get("useAlphaTest", true);
    }

    override getProperties(): Properties {
        return new Properties({
            lightIndex: this.lightIndex,
            cascadeCount: this.cascadeCount,
            mapSize: this.mapSize,
            splitLambda: this.splitLambda,
            maxDistance: this.maxDistance,
            useAlphaTest: this.useAlphaTest,
        });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Cascades", this.cascadeCount, 1, kMaxCascades, 1, (v) => {
            this.cascadeCount = Math.round(v);
            this.requestRecompile();
        });
        ui.dropdown("Map size", ["512", "1024", "2048", "4096"], String(this.mapSize), (v) => {
            this.mapSize = Number(v);
            this.requestRecompile();
        });
        ui.slider("Split lambda", this.splitLambda, 0, 1, 0.01, (v) => (this.splitLambda = v));
        ui.slider("Max distance (0 = scene)", this.maxDistance, 0, 1000, 0.1, (v) => (this.maxDistance = v));
        ui.checkbox("Alpha test", this.useAlphaTest, (v) => {
            this.useAlphaTest = v;
            this.state = null;
        });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("shadowMap", "Light-space depth atlas (R32F, cascades side by side)")
            .texture2D(this.mapSize * this.cascadeCount, this.mapSize)
            .format(ResourceFormat.R32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.state = null;
        this.vao = null;
        this.warned = false;
    }

    /** Index into the scene's active (GPU) light list of the shadowed directional light, or -1. */
    private findLight(): number {
        const lights = this.scene!.activeLights;
        if (this.lightIndex >= 0) return this.lightIndex < lights.length && lights[this.lightIndex]!.type === LightType.Directional ? this.lightIndex : -1;
        return lights.findIndex((l) => l.type === LightType.Directional);
    }

    private build(): void {
        const scene = this.scene!;
        const defines = scene.getSceneDefines();
        defines.add("USE_ALPHA_TEST", this.useAlphaTest ? 1 : 0);
        const program = this.device.programManager.createProgram(
            { path: kShaderFile, entryPoints: [{ name: "vsMain", type: ShaderType.Vertex }, { name: "psMain", type: ShaderType.Pixel }] },
            defines,
        );
        const version = program.getActiveVersion();
        const vs = version.getKernel("vsMain", ShaderType.Vertex);
        const ps = version.getKernel("psMain", ShaderType.Pixel);
        const bindings = mergeWgslBindings(vs.bindings, ps.bindings);
        // One parameter block per cascade: per-cascade constants must not alias within one submit.
        this.cascades = [];
        for (let c = 0; c < kMaxCascades; c++) {
            const vars = new ParameterBlock(this.device, version.reflection, bindings);
            this.cascades.push({ vars, root: makeRootVar(vars) });
        }
        const state = new GraphicsState(this.device).setKernels(vs, ps);
        state.setVao(this.getVao());
        state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
        state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc()));
        const vars0 = this.cascades[0]!.vars;
        const groupIndices = vars0.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) layouts.push(vars0.getBindGroupLayout(g) ?? this.device.gpuDevice.createBindGroupLayout({ entries: [] }));
        this.pipelineLayout = this.device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts });
        this.state = state;
    }

    private getVao(): Vao {
        if (this.vao) return this.vao;
        const vertexLayout = new VertexLayout();
        const vb = new VertexBufferLayout();
        vb.addElement("POSITION", 0, ResourceFormat.RGB32Float, 1, 0);
        vb.addElement("PACKED_NORMAL_TANGENT_CURVE_RADIUS", 16, ResourceFormat.RGB32Float, 1, 1);
        vb.addElement("TEXCOORD", 32, ResourceFormat.RG32Float, 1, 2);
        vb.stride = 48;
        const ib = new VertexBufferLayout();
        ib.addElement("DRAW_ID", 0, ResourceFormat.R32Uint, 1, 3);
        ib.stride = 4;
        ib.setInputClass(InputClass.PerInstanceData, 1);
        vertexLayout.addBufferLayout(0, vb).addBufferLayout(1, ib);
        const drawData = this.scene!.getMeshDrawData();
        this.vao = new Vao(Topology.TriangleList, vertexLayout, [drawData.vertexBuffer, drawData.drawIDBuffer], drawData.indexBuffer, ResourceFormat.R32Uint);
        return this.vao;
    }

    /** Fits the cascades to the current camera and light (CPU, double precision). */
    computeCascades(lightDir: V3): CascadedShadowMapData {
        const scene = this.scene!;
        const cam = scene.camera;
        const pos = v3(cam.getPosition());
        const fwd = norm(sub(v3(cam.getTarget()), pos));
        const right = norm(cross(fwd, v3(cam.getUpVector())));
        const up = cross(right, fwd);
        const tanY = Math.tan(cam.getFovY() / 2);
        const tanX = tanY * cam.getAspectRatio();

        // Scene box corners (for the view range and the caster depth range).
        const b = scene.worldBounds;
        const corners: V3[] = [];
        if (b) for (let i = 0; i < 8; i++) corners.push([i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]]);
        let near = Math.max(cam.getNearPlane(), 1e-3);
        let far = cam.getFarPlane();
        if (corners.length) {
            const depths = corners.map((p) => dot(sub(p, pos), fwd));
            far = Math.min(far, Math.max(...depths));
            near = Math.max(near, Math.min(...depths));
            if (!(far > near)) far = near * 2;
        }
        if (this.maxDistance > 0) far = Math.min(far, this.maxDistance);

        const n = this.cascadeCount;
        const splits: number[] = [];
        for (let i = 1; i <= n; i++) {
            const log = near * Math.pow(far / near, i / n);
            const lin = near + ((far - near) * i) / n;
            splits.push(this.splitLambda * log + (1 - this.splitLambda) * lin);
        }

        const f = norm(lightDir);
        const up0: V3 = Math.abs(f[1]) < 0.99 ? [0, 1, 0] : [1, 0, 0];
        const lr = norm(cross(f, up0));
        const lu = cross(lr, f);
        const S = this.mapSize;
        const out: CascadedShadowMapData = { lightIndex: -1, cascadeCount: n, size: S, viewProj: [], splits, texelWorld: [], depthRange: [] };
        let z0 = near;
        for (let c = 0; c < n; c++) {
            const z1 = splits[c]!;
            // Bounding sphere of the frustum slice [z0, z1].
            const pts: V3[] = [];
            for (const z of [z0, z1]) {
                const cz = add(pos, mul(fwd, z));
                for (const sx of [-1, 1]) for (const sy of [-1, 1]) pts.push(add(cz, add(mul(right, sx * z * tanX), mul(up, sy * z * tanY))));
            }
            let center: V3 = [0, 0, 0];
            for (const p of pts) center = add(center, mul(p, 1 / pts.length));
            let radius = 0;
            for (const p of pts) radius = Math.max(radius, Math.hypot(...sub(p, center)));
            radius = Math.ceil(radius * 16) / 16; // quantized: stable texel size under camera rotation
            // Texel snapping of the box centre in light space (no shimmering under camera translation).
            const texel = (2 * radius) / S;
            const cr = Math.round(dot(center, lr) / texel) * texel;
            const cu = Math.round(dot(center, lu) / texel) * texel;
            // Depth range: every scene caster plus the slice.
            let dmin = dot(center, f) - radius;
            let dmax = dot(center, f) + radius;
            for (const p of corners) {
                dmin = Math.min(dmin, dot(p, f));
                dmax = Math.max(dmax, dot(p, f));
            }
            const range = Math.max(dmax - dmin, 1e-4);
            dmin -= range * 1e-3;
            const rangeP = (dmax - dmin) * (1 + 1e-3);
            const m = new float4x4();
            const set = (row: number, a: V3, s: number, o: number) => {
                m.data[row * 4 + 0] = a[0] * s;
                m.data[row * 4 + 1] = a[1] * s;
                m.data[row * 4 + 2] = a[2] * s;
                m.data[row * 4 + 3] = o;
            };
            set(0, lr, 1 / radius, -cr / radius);
            set(1, lu, 1 / radius, -cu / radius);
            set(2, f, 1 / rangeP, -dmin / rangeP);
            m.data[15] = 1;
            out.viewProj.push(m);
            out.texelWorld.push(texel);
            out.depthRange.push(rangeP);
            z0 = z1;
        }
        return out;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const atlas = renderData.getTexture("shadowMap")!;
        ctx.clearTexture(atlas, [1, 1, 1, 1]);
        renderData.dictionary.delete(kCascadedShadowMapData);
        if (!this.scene) return;
        const li = this.findLight();
        if (li < 0) {
            if (!this.warned) Logger.warning("CascadedShadowMap: the scene has no (active) directional light; the shadow map stays empty.");
            this.warned = true;
            return;
        }
        const light = this.scene.activeLights[li]!;
        const data = this.computeCascades(v3(light.dirW!));
        data.lightIndex = li;
        renderData.dictionary.set(kCascadedShadowMapData, data);

        if (!this.state) this.build();
        if (!this.depth || this.depth.width !== atlas.width || this.depth.height !== atlas.height) {
            this.depth?.destroy();
            this.depth = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: atlas.width,
                height: atlas.height,
                format: ResourceFormat.D32Float,
                bindFlags: ResourceBindFlags.DepthStencil,
                name: "CascadedShadowMap::depth",
            });
        }
        ctx.clearDsv(this.depth.getDSV(), 1, 0);

        const fbo = new Fbo();
        fbo.attachColorTarget(atlas, 0);
        fbo.attachDepthStencilTarget(this.depth);
        const state = this.state!;
        state.setFbo(fbo);
        const gso = state.getGSO(this.pipelineLayout!);
        const vao = state.getVao()!;
        const draws = this.scene.getMeshDrawData().draws;

        // Bind groups first: dirty cbuffers upload on the encoder, which is locked while a render pass is open.
        const groups = [];
        for (let c = 0; c < data.cascadeCount; c++) {
            const { vars, root } = this.cascades[c]!;
            this.scene.bindShaderData(root);
            root["PerCascadeCB"]["gLightViewProj"] = data.viewProj[c]!;
            groups.push(vars.getGroupIndices().map((g) => ({ index: g, group: vars.getBindGroup(g) })));
        }

        const desc = fbo.getGpuRenderPassDescriptor();
        const tw = this.device.profilerHook?.passTimestampWrites();
        if (tw) desc.timestampWrites = tw;
        const pass = ctx.getEncoder().beginRenderPass(desc);
        pass.setPipeline(gso.gpuPipeline);
        vao.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        pass.setIndexBuffer(vao.indexBuffer!.gpuBuffer, vao.getGpuIndexFormat());
        for (let c = 0; c < data.cascadeCount; c++) {
            for (const { index, group } of groups[c]!) pass.setBindGroup(index, group);
            pass.setViewport(c * data.size, 0, data.size, data.size, 0, 1);
            for (const draw of draws) pass.drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.baseVertex, draw.firstInstance);
        }
        pass.end();
    }
}

registerRenderPass("CascadedShadowMap", (device, props) => new CascadedShadowMap(device, props), {
    desc: "Cascaded shadow map (R32F atlas) for a directional light",
});
