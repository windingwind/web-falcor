/**
 * Raster G-buffer pass mirroring Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.
 * Uses the upstream GBufferRaster.3d.slang (via WebFalcor override) with the
 * scene's define set; program is created lazily once a scene is bound (as
 * upstream does in execute()).
 */

import {
    DefineList,
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
    makeRootVar,
    mergeWgslBindings,
    registerRenderPass,
    type CompileData,
    type Device,
    type ProgramVersion,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang";

/** SV_TARGET order and formats (kGBufferChannels in GBuffer.cpp). */
const kChannels: { name: string; format: ResourceFormat }[] = [
    { name: "posW", format: ResourceFormat.RGBA32Float },
    { name: "normW", format: ResourceFormat.RGBA32Float },
    { name: "tangentW", format: ResourceFormat.RGBA32Float },
    { name: "faceNormalW", format: ResourceFormat.RGBA32Float },
    { name: "texC", format: ResourceFormat.RG32Float },
    { name: "texGrads", format: ResourceFormat.RGBA16Float },
    { name: "mvec", format: ResourceFormat.RG32Float },
    { name: "mtlData", format: ResourceFormat.RGBA32Uint },
];

export class GBufferRaster extends RenderPass {
    private version: ProgramVersion | null = null;
    private vars: ParameterBlock | null = null;
    private root: ShaderVar | null = null;
    private state: GraphicsState | null = null;
    private pipelineLayout: GPUPipelineLayout | null = null;
    private depthTex: Texture | null = null;

    constructor(device: Device, _props: Properties) {
        super(device);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        for (const ch of kChannels) {
            r.addOutput(ch.name, `G-buffer ${ch.name}`)
                .texture2D(w, h)
                .format(ch.format)
                .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        }
        r.addOutput("depth", "Depth buffer")
            .texture2D(w, h)
            .format(ResourceFormat.D32Float)
            .bindFlags(ResourceBindFlags.DepthStencil | ResourceBindFlags.ShaderResource);
        return r;
    }

    private createProgram(): void {
        const scene = this.scene!;
        const defines = scene.getSceneDefines();
        // Channel validity: raster targets on, UAV/extra channels off (v1).
        for (const name of ["gPosW", "gNormW", "gTangentW", "gFaceNormalW", "gTexC", "gMaterialData"]) defines.add(`is_valid_${name}`, 1);
        for (const name of ["gTexGrads", "gMotionVector", "gViewW", "gGuideNormalW", "gDiffOpacity", "gSpecRough", "gEmissive", "gMask", "gPosNormalFwidth", "gLinearZAndDeriv", "gVBuffer", "gDepth", "gRoughness", "gDisocclusion"]) defines.add(`is_valid_${name}`, 0);
        defines.add("USE_ALPHA_TEST", 0);
        defines.add("ADJUST_SHADING_NORMALS", 0);

        const program = this.device.programManager.createProgram(
            {
                path: kShaderFile,
                entryPoints: [
                    { name: "vsMain", type: ShaderType.Vertex },
                    { name: "psMain", type: ShaderType.Pixel },
                ],
            },
            defines,
        );
        this.version = program.getActiveVersion();
        const vs = this.version.getKernel("vsMain", ShaderType.Vertex);
        const ps = this.version.getKernel("psMain", ShaderType.Pixel);
        this.vars = new ParameterBlock(this.device, this.version.reflection, mergeWgslBindings(vs.bindings, ps.bindings));
        this.root = makeRootVar(this.vars);

        // Vertex layout: packed vertex buffer (48B stride) + per-instance draw IDs.
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

        const drawData = scene.getMeshDrawData();
        const vao = new Vao(Topology.TriangleList, vertexLayout, [drawData.vertexBuffer, drawData.drawIDBuffer], drawData.indexBuffer, ResourceFormat.R32Uint);

        this.state = new GraphicsState(this.device).setKernels(vs, ps);
        this.state.setVao(vao);
        // v1: no culling (asset winding conventions handled with SceneBuilder flags later).
        this.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
        this.state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc()));

        const groupIndices = this.vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) {
            layouts.push(this.vars.getBindGroupLayout(g) ?? this.device.gpuDevice.createBindGroupLayout({ entries: [] }));
        }
        this.pipelineLayout = this.device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts });
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.version = null; // program depends on scene defines
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        if (!this.version) this.createProgram();

        const fbo = new Fbo();
        kChannels.forEach((ch, i) => fbo.attachColorTarget(renderData.getTexture(ch.name)!, i));
        fbo.attachDepthStencilTarget(renderData.getTexture("depth")!);

        // Clear all targets + depth (mirrors GBufferRaster's clearing of RTs).
        for (const ch of kChannels) ctx.clearTexture(renderData.getTexture(ch.name)!, [0, 0, 0, 0]);
        ctx.clearDsv(renderData.getTexture("depth")!.getDSV(), 1, 0);

        this.scene.bindShaderData(this.root!);

        const state = this.state!;
        state.setFbo(fbo);
        const gso = state.getGSO(this.pipelineLayout!);
        const vao = state.getVao()!;
        const vars = this.vars!;
        const bindGroups = vars.getGroupIndices().map((g) => ({ index: g, group: vars.getBindGroup(g) }));

        const pass = ctx.getEncoder().beginRenderPass(fbo.getGpuRenderPassDescriptor());
        pass.setPipeline(gso.gpuPipeline);
        pass.setViewport(0, 0, fbo.width, fbo.height, 0, 1);
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        vao.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        pass.setIndexBuffer(vao.indexBuffer!.gpuBuffer, vao.getGpuIndexFormat());
        for (const draw of this.scene.getMeshDrawData().draws) {
            pass.drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.baseVertex, draw.firstInstance);
        }
        pass.end();
    }
}

registerRenderPass("GBufferRaster", (device, props) => new GBufferRaster(device, props));
