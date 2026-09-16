/**
 * Raster V-buffer pass mirroring Source/RenderPasses/GBuffer/VBuffer/
 * VBufferRaster. The override shader draws non-indexed with scene vertex
 * pulling (WGSL has no fragment barycentrics/primitive id, docs §9); the
 * optional mvec/mask channels are extra render targets (no ROVs on WebGPU).
 */

import {
    DepthStencilState,
    DepthStencilStateDesc,
    Fbo,
    FieldFlags,
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
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang";

/** Optional extra channels (kVBufferExtraChannels): [output name, shader texture name, format]. */
const kExtraChannels: { name: string; texname: string; format: ResourceFormat; desc: string }[] = [
    { name: "mvec", texname: "gMotionVector", format: ResourceFormat.RG32Float, desc: "Motion vector" },
    { name: "mask", texname: "gMask", format: ResourceFormat.R32Float, desc: "Mask" },
];

export class VBufferRaster extends RenderPass {
    private version: ProgramVersion | null = null;
    private vars: ParameterBlock | null = null;
    private root: ShaderVar | null = null;
    private state: GraphicsState | null = null;
    private pipelineLayout: GPUPipelineLayout | null = null;
    /** Extra-channel validity the current program was compiled for (is_valid_ defines). */
    private programKey = "";
    private outputSize = IOSize.Default;
    /** Native kFixedOutputSize default (used when outputSize == Fixed). */
    private fixedOutputSize: [number, number] = [512, 512];
    private sampleCount = 16;
    private sampleGenerator: CPUSampleGenerator | null = null;
    private useAlphaTest = true;
    private samplePattern = "Center";
    /** Native GBufferBase forceCullMode/cull; web default without forcing is None (raster == software-RT coverage). */
    private forceCullMode = false;
    private cullMode = CullMode.Back;

    constructor(device: Device, props: Properties) {
        super(device);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
        this.sampleCount = props.get("sampleCount", 16);
        this.useAlphaTest = props.get("useAlphaTest", true);
        this.samplePattern = props.get<string>("samplePattern", "Center");
        this.forceCullMode = props.get("forceCullMode", false);
        const cull = props.getOpt<string | number>("cull");
        if (cull !== undefined) this.cullMode = (typeof cull === "string" ? CullMode[cull as keyof typeof CullMode] : cull) ?? CullMode.Back;
        this.updateSamplePattern();
    }

    /** Mirrors GBufferBase::updateSamplePattern (Center -> no generator). */
    private updateSamplePattern(): void {
        const c = this.sampleCount;
        this.sampleGenerator =
            this.samplePattern === "Stratified" ? new StratifiedSamplePattern(c)
            : this.samplePattern === "Halton" ? new HaltonSamplePattern(c)
            : this.samplePattern === "DirectX" ? new DxSamplePattern(c)
            : null;
        if (this.sampleGenerator) this.sampleCount = this.sampleGenerator.getSampleCount();
    }

    override getProperties(): Properties {
        return new Properties({ outputSize: IOSize[this.outputSize]!, fixedOutputSize: this.fixedOutputSize, samplePattern: this.samplePattern, sampleCount: this.sampleCount, useAlphaTest: this.useAlphaTest, forceCullMode: this.forceCullMode, cull: CullMode[this.cullMode]! });
    }

    /** Mirrors GBufferBase::renderUI (alpha test is a define -> program rebuild; output size ⏳). */
    override renderUI(ui: UIWidgets): void {
        // Native GBufferBase/ImageLoader/ToneMapper/... "Output size" controls: I/O size changes recompile the graph.
        ui.dropdown("Output size", ["Default", "Fixed", "Full", "Half", "Quarter", "Double"], IOSize[this.outputSize]!, (v) => {
            this.outputSize = IOSize[v as keyof typeof IOSize];
            this.requestRecompile();
        });
        ui.slider("Size in pixels (width)", this.fixedOutputSize[0], 32, 4096, 1, (v) => {
            this.fixedOutputSize = [Math.round(v), this.fixedOutputSize[1]];
            this.requestRecompile();
        });
        ui.slider("Size in pixels (height)", this.fixedOutputSize[1], 32, 4096, 1, (v) => {
            this.fixedOutputSize = [this.fixedOutputSize[0], Math.round(v)];
            this.requestRecompile();
        });
        ui.checkbox("Force cull mode", this.forceCullMode, (v) => {
            this.forceCullMode = v;
            this.version = null;
        });
        ui.dropdown("Cull mode", ["None", "Front", "Back"], CullMode[this.cullMode]!, (v) => {
            this.cullMode = CullMode[v as keyof typeof CullMode];
            this.version = null;
        });
        ui.dropdown("Sample pattern", ["Center", "DirectX", "Halton", "Stratified"], this.samplePattern, (v) => {
            this.samplePattern = v;
            this.updateSamplePattern();
        });
        ui.slider("Sample count", this.sampleCount, 1, 1024, 1, (v) => {
            this.sampleCount = Math.max(1, Math.round(v));
            this.updateSamplePattern();
        });
        ui.checkbox("Alpha Test", this.useAlphaTest, (v) => {
            this.useAlphaTest = v;
            this.version = null;
        });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
        r.addOutput("vbuffer", "V-buffer in packed format (indices + barycentrics)")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Uint)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        r.addOutput("depth", "Depth buffer")
            .texture2D(w, h)
            .format(ResourceFormat.D32Float)
            .bindFlags(ResourceBindFlags.DepthStencil | ResourceBindFlags.ShaderResource);
        // Mirrors addRenderPassOutputs(kVBufferExtraChannels): optional, allocated when consumed.
        for (const ch of kExtraChannels) {
            r.addOutput(ch.name, ch.desc)
                .texture2D(w, h)
                .format(ch.format)
                .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        return r;
    }

    private createProgram(valid: Record<string, number>): void {
        const scene = this.scene!;
        const defines = scene.getSceneDefines();
        defines.add("USE_ALPHA_TEST", this.useAlphaTest ? 1 : 0);
        defines.addAll(valid);

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
        this.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(this.forceCullMode ? this.cullMode : CullMode.None)));
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
        const vbuffer = renderData.getTexture("vbuffer")!;
        const extras = kExtraChannels.map((ch) => renderData.getTexture(ch.name));
        // Mirrors VBufferRaster::execute: clear outputs (and the optional channels) before drawing.
        ctx.clearTexture(vbuffer, [0, 0, 0, 0]);
        ctx.clearDsv(renderData.getTexture("depth")!.getDSV(), 1, 0);
        extras.forEach((tex) => tex && ctx.clearTexture(tex, [0, 0, 0, 0]));
        if (!this.scene) return;

        // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
        const valid: Record<string, number> = {};
        kExtraChannels.forEach((ch, i) => (valid[`is_valid_${ch.texname}`] = extras[i] ? 1 : 0));
        const key = JSON.stringify(valid);
        if (!this.version || this.programKey !== key) {
            this.createProgram(valid);
            this.programKey = key;
        }

        this.scene.camera.setPatternGenerator(
            this.sampleGenerator,
            new float2(Math.fround(1 / vbuffer.width), Math.fround(1 / vbuffer.height)),
        );

        const fbo = new Fbo();
        fbo.attachColorTarget(vbuffer, 0);
        extras.forEach((tex, i) => tex && fbo.attachColorTarget(tex, i + 1));
        fbo.attachDepthStencilTarget(renderData.getTexture("depth")!);

        this.scene.bindShaderData(this.root!);
        this.root!["PerFrameCB"]["gFrameDim"] = [vbuffer.width, vbuffer.height];

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
