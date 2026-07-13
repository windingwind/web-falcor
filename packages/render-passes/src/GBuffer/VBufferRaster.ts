/**
 * Raster V-buffer pass mirroring Source/RenderPasses/GBuffer/VBuffer/
 * VBufferRaster. The override shader draws non-indexed with scene vertex
 * pulling (WGSL has no fragment barycentrics/primitive id, docs §9);
 * mvec/mask extra channels are not produced.
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
    ShaderType,
    Vao,
    VertexBufferLayout,
    VertexLayout,
    Topology,
    InputClass,
    IOSize,
    parseIOSize,
    calculateIOSize,
    StratifiedSamplePattern,
    HaltonSamplePattern,
    DxSamplePattern,
    float2,
    makeRootVar,
    mergeWgslBindings,
    registerRenderPass,
    type CompileData,
    type CPUSampleGenerator,
    type Device,
    type ProgramVersion,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang";

export class VBufferRaster extends RenderPass {
    private version: ProgramVersion | null = null;
    private vars: ParameterBlock | null = null;
    private root: ShaderVar | null = null;
    private state: GraphicsState | null = null;
    private pipelineLayout: GPUPipelineLayout | null = null;
    private outputSize = IOSize.Default;
    private sampleCount = 16;
    private sampleGenerator: CPUSampleGenerator | null = null;
    private useAlphaTest = true;

    constructor(device: Device, props: Properties) {
        super(device);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        this.sampleCount = props.get("sampleCount", 16);
        this.useAlphaTest = props.get("useAlphaTest", true);
        const pattern = props.get<string>("samplePattern", "Center");
        if (pattern === "Stratified") this.sampleGenerator = new StratifiedSamplePattern(this.sampleCount);
        else if (pattern === "Halton") this.sampleGenerator = new HaltonSamplePattern(this.sampleCount);
        else if (pattern === "DirectX") this.sampleGenerator = new DxSamplePattern(this.sampleCount);
        if (this.sampleGenerator) this.sampleCount = this.sampleGenerator.getSampleCount();
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, [512, 512], compileData.defaultTexDims);
        r.addOutput("vbuffer", "V-buffer in packed format (indices + barycentrics)")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Uint)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        r.addOutput("depth", "Depth buffer")
            .texture2D(w, h)
            .format(ResourceFormat.D32Float)
            .bindFlags(ResourceBindFlags.DepthStencil | ResourceBindFlags.ShaderResource);
        return r;
    }

    private createProgram(): void {
        const scene = this.scene!;
        const defines = scene.getSceneDefines();
        defines.add("USE_ALPHA_TEST", this.useAlphaTest ? 1 : 0);
        for (const name of ["gMotionVector", "gMask"]) defines.add(`is_valid_${name}`, 0);

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

        // Only the per-instance draw-ID stream; geometry is vertex-pulled.
        const vertexLayout = new VertexLayout();
        const ib = new VertexBufferLayout();
        ib.addElement("DRAW_ID", 0, ResourceFormat.R32Uint, 1, 0);
        ib.stride = 4;
        ib.setInputClass(InputClass.PerInstanceData, 1);
        vertexLayout.addBufferLayout(0, ib);

        const drawData = scene.getMeshDrawData();
        const vao = new Vao(Topology.TriangleList, vertexLayout, [drawData.drawIDBuffer]);

        this.state = new GraphicsState(this.device).setKernels(vs, ps);
        this.state.setVao(vao);
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
        this.version = null;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        if (!this.version) this.createProgram();

        const vbuffer = renderData.getTexture("vbuffer")!;
        this.scene.camera.setPatternGenerator(
            this.sampleGenerator,
            new float2(Math.fround(1 / vbuffer.width), Math.fround(1 / vbuffer.height)),
        );

        const fbo = new Fbo();
        fbo.attachColorTarget(vbuffer, 0);
        fbo.attachDepthStencilTarget(renderData.getTexture("depth")!);
        ctx.clearTexture(vbuffer, [0, 0, 0, 0]);
        ctx.clearDsv(renderData.getTexture("depth")!.getDSV(), 1, 0);

        this.scene.bindShaderData(this.root!);

        const state = this.state!;
        state.setFbo(fbo);
        const gso = state.getGSO(this.pipelineLayout!);
        const vao = state.getVao()!;
        const vars = this.vars!;
        const bindGroups = vars.getGroupIndices().map((g) => ({ index: g, group: vars.getBindGroup(g) }));

        const desc = fbo.getGpuRenderPassDescriptor();
        const tw = this.device.profilerHook?.passTimestampWrites();
        if (tw) desc.timestampWrites = tw;
        const pass = ctx.getEncoder().beginRenderPass(desc);
        pass.setPipeline(gso.gpuPipeline);
        pass.setViewport(0, 0, fbo.width, fbo.height, 0, 1);
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        vao.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        // Non-indexed: vertexID counts 0..3*triCount per draw (vertex pulling).
        for (const draw of this.scene.getMeshDrawData().draws) {
            pass.draw(draw.indexCount, 1, 0, draw.firstInstance);
        }
        pass.end();
    }
}

registerRenderPass("VBufferRaster", (device, props) => new VBufferRaster(device, props));
