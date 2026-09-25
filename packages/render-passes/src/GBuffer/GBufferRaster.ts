/**
 * Raster G-buffer pass mirroring Source/RenderPasses/GBuffer/GBuffer/GBufferRaster.
 * Uses the upstream GBufferRaster.3d.slang (via WebFalcor override) with the
 * scene's define set; programs are created lazily once a scene is bound (as
 * upstream does in execute()). WebGPU has no rasterizer-ordered views, so the
 * upstream UAV "extra" channels are render targets here: connected channels are
 * batched into groups of <= 8 targets and the geometry is drawn once per batch
 * (docs §9). The vbuffer channel needs SV_PrimitiveID: when it is connected the
 * pass switches to non-indexed vertex pulling (WEBFALCOR_VERTEX_PULLING), like
 * VBufferRaster, so the triangle index and barycentrics are available.
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
    ResourceType,
    ShaderType,
    Texture,
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
    type RenderContext,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang";

/** [output name, shader texture name, format] — kGBufferChannels followed by kGBufferExtraChannels (GBufferRaster.cpp). */
const kChannels: { name: string; texname: string; format: ResourceFormat; desc: string }[] = [
    { name: "posW", texname: "gPosW", format: ResourceFormat.RGBA32Float, desc: "World space position" },
    { name: "normW", texname: "gNormW", format: ResourceFormat.RGBA32Float, desc: "World space normal" },
    { name: "tangentW", texname: "gTangentW", format: ResourceFormat.RGBA32Float, desc: "World space tangent" },
    { name: "faceNormalW", texname: "gFaceNormalW", format: ResourceFormat.RGBA32Float, desc: "Face normal in world space" },
    { name: "texC", texname: "gTexC", format: ResourceFormat.RG32Float, desc: "Texture coordinate" },
    { name: "texGrads", texname: "gTexGrads", format: ResourceFormat.RGBA16Float, desc: "Texture gradients (ddx, ddy)" },
    { name: "mvec", texname: "gMotionVector", format: ResourceFormat.RG32Float, desc: "Motion vector" },
    { name: "mtlData", texname: "gMaterialData", format: ResourceFormat.RGBA32Uint, desc: "Material data" },
    { name: "guideNormalW", texname: "gGuideNormalW", format: ResourceFormat.RGBA32Float, desc: "Guide normal in world space" },
    { name: "diffuseOpacity", texname: "gDiffOpacity", format: ResourceFormat.RGBA32Float, desc: "Diffuse reflection albedo and opacity" },
    { name: "specRough", texname: "gSpecRough", format: ResourceFormat.RGBA32Float, desc: "Specular reflectance and roughness" },
    { name: "emissive", texname: "gEmissive", format: ResourceFormat.RGBA32Float, desc: "Emissive color" },
    { name: "viewW", texname: "gViewW", format: ResourceFormat.RGBA32Float, desc: "View direction in world space" },
    { name: "pnFwidth", texname: "gPosNormalFwidth", format: ResourceFormat.RG32Float, desc: "Position and guide normal filter width" },
    { name: "linearZ", texname: "gLinearZAndDeriv", format: ResourceFormat.RG32Float, desc: "Linear z (and derivative)" },
    { name: "mask", texname: "gMask", format: ResourceFormat.R32Float, desc: "Mask" },
    { name: "vbuffer", texname: "gVBuffer", format: ResourceFormat.RGBA32Uint, desc: "Visibility buffer" },
];
type Channel = (typeof kChannels)[number];

/** WebGPU guarantees 8 color attachments; each batch is one geometry pass. */
const kMaxTargetsPerBatch = 8;

/** One compiled program + state per connected-channel batch (and option set). */
interface Variant {
    vars: ParameterBlock;
    root: ShaderVar;
    state: GraphicsState;
    pipelineLayout: GPUPipelineLayout;
}

export class GBufferRaster extends RenderPass {
    private variants = new Map<string, Variant>();
    private vao: Vao | null = null;
    private pullVao: Vao | null = null;
    /** Stand-in color target when only the depth output is consumed. */
    private scratchTarget: Texture | null = null;
    private outputSize = IOSize.Default;
    /** Native kFixedOutputSize default (used when outputSize == Fixed). */
    private fixedOutputSize: [number, number] = [512, 512];
    private sampleCount = 16;
    private sampleGenerator: CPUSampleGenerator | null = null;
    private samplePattern = "Center";
    private useAlphaTest = true;
    private adjustShadingNormals = true;
    /** Native GBufferBase forceCullMode/cull; web default without forcing is None (raster == software-RT coverage). */
    private forceCullMode = false;
    private cullMode = CullMode.Back;

    constructor(device: Device, props: Properties) {
        super(device);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fixed = props.getOpt<number[] | { x: number; y: number }>("fixedOutputSize");
        if (fixed) this.fixedOutputSize = Array.isArray(fixed) ? [fixed[0]!, fixed[1]!] : [fixed.x, fixed.y];
        this.sampleCount = props.get("sampleCount", 16);
        this.samplePattern = props.get<string>("samplePattern", "Center");
        this.useAlphaTest = props.get("useAlphaTest", true);
        this.adjustShadingNormals = props.get("adjustShadingNormals", true);
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
        return new Properties({
            outputSize: IOSize[this.outputSize]!,
            fixedOutputSize: this.fixedOutputSize,
            samplePattern: this.samplePattern,
            sampleCount: this.sampleCount,
            useAlphaTest: this.useAlphaTest,
            adjustShadingNormals: this.adjustShadingNormals,
            forceCullMode: this.forceCullMode,
            cull: CullMode[this.cullMode]!,
        });
    }

    /** Mirrors GBufferBase::renderUI. */
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
        ui.dropdown("Sample pattern", ["Center", "DirectX", "Halton", "Stratified"], this.samplePattern, (v) => {
            this.samplePattern = v;
            this.updateSamplePattern();
        });
        ui.slider("Sample count", this.sampleCount, 1, 1024, 1, (v) => {
            this.sampleCount = Math.max(1, Math.round(v));
            this.updateSamplePattern();
        });
        const rebuild = <T>(set: (v: T) => void) => (v: T) => {
            set(v);
            this.variants.clear();
        };
        ui.checkbox("Alpha Test", this.useAlphaTest, rebuild((v) => (this.useAlphaTest = v)));
        ui.checkbox("Adjust shading normals", this.adjustShadingNormals, rebuild((v) => (this.adjustShadingNormals = v)));
        ui.checkbox("Force cull mode", this.forceCullMode, rebuild((v) => (this.forceCullMode = v)));
        ui.dropdown("Cull mode", ["None", "Front", "Back"], CullMode[this.cullMode]!, rebuild((v: string) => (this.cullMode = CullMode[v as keyof typeof CullMode])));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = calculateIOSize(this.outputSize, this.fixedOutputSize, compileData.defaultTexDims);
        // Native: every channel is optional; the web renders the extra (UAV) channels as targets too.
        for (const ch of kChannels) {
            r.addOutput(ch.name, ch.desc)
                .texture2D(w, h)
                .format(ch.format)
                .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource)
                .flags(FieldFlags.Optional);
        }
        r.addOutput("depth", "Depth buffer")
            .texture2D(w, h)
            .format(ResourceFormat.D32Float)
            .bindFlags(ResourceBindFlags.DepthStencil | ResourceBindFlags.ShaderResource);
        return r;
    }

    /** Compiles the program for one batch of channels (is_valid_/LOC_ defines pick the targets). */
    private getVariant(batch: Channel[], pulling: boolean): Variant {
        const key = `${batch.map((c) => c.texname).join(",")}|${pulling}|${this.useAlphaTest}|${this.adjustShadingNormals}|${this.forceCullMode ? this.cullMode : -1}`;
        const cached = this.variants.get(key);
        if (cached) return cached;

        const scene = this.scene!;
        const defines = scene.getSceneDefines();
        for (const ch of kChannels) {
            const loc = batch.indexOf(ch);
            defines.add(`is_valid_${ch.texname}`, loc >= 0 ? 1 : 0);
            defines.add(`LOC_${ch.texname}`, Math.max(loc, 0));
        }
        defines.add("WEBFALCOR_VERTEX_PULLING", pulling ? 1 : 0);
        defines.add("USE_ALPHA_TEST", this.useAlphaTest ? 1 : 0);
        defines.add("ADJUST_SHADING_NORMALS", this.adjustShadingNormals ? 1 : 0);

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
        const version = program.getActiveVersion();
        const vs = version.getKernel("vsMain", ShaderType.Vertex);
        const ps = version.getKernel("psMain", ShaderType.Pixel);
        const vars = new ParameterBlock(this.device, version.reflection, mergeWgslBindings(vs.bindings, ps.bindings));
        const root = makeRootVar(vars);

        const state = new GraphicsState(this.device).setKernels(vs, ps);
        state.setVao(pulling ? this.getPullVao() : this.getVao());
        // Web default = no culling so raster coverage equals the software-RT passes (native default: Back); forceCullMode overrides.
        state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(this.forceCullMode ? this.cullMode : CullMode.None)));
        state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc()));

        const groupIndices = vars.getGroupIndices();
        const maxGroup = groupIndices.length ? Math.max(...groupIndices) : -1;
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= maxGroup; g++) {
            layouts.push(vars.getBindGroupLayout(g) ?? this.device.gpuDevice.createBindGroupLayout({ entries: [] }));
        }
        const pipelineLayout = this.device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts });
        const variant = { vars, root, state, pipelineLayout };
        this.variants.set(key, variant);
        return variant;
    }

    /** Vertex layout: packed vertex buffer (48B stride) + per-instance draw IDs. */
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

    /** Vertex-pulling layout: only the per-instance draw-ID stream (geometry comes from scene buffers). */
    private getPullVao(): Vao {
        if (this.pullVao) return this.pullVao;
        const vertexLayout = new VertexLayout();
        const ib = new VertexBufferLayout();
        ib.addElement("DRAW_ID", 0, ResourceFormat.R32Uint, 1, 0);
        ib.stride = 4;
        ib.setInputClass(InputClass.PerInstanceData, 1);
        vertexLayout.addBufferLayout(0, ib);
        this.pullVao = new Vao(Topology.TriangleList, vertexLayout, [this.scene!.getMeshDrawData().drawIDBuffer]);
        return this.pullVao;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        // GBufferBase::setScene: a fresh jitter pattern (the stratified one carries RNG state).
        this.updateSamplePattern();
        this.variants.clear(); // programs depend on scene defines
        this.vao = null;
        this.pullVao = null;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const depth = renderData.getTexture("depth")!;
        const [w, h] = [depth.width, depth.height];
        const present = kChannels.filter((ch) => renderData.getTexture(ch.name) !== undefined);

        // Mirrors GBufferRaster::execute: clear depth and all connected channels first.
        ctx.clearDsv(depth.getDSV(), 1, 0);
        for (const ch of present) ctx.clearTexture(renderData.getTexture(ch.name)!, [0, 0, 0, 0]);
        if (!this.scene) return;

        // Mirrors GBufferBase::updateFrameDim: the camera jitters by the sample
        // pattern scaled to the pass resolution (first sample lands next frame).
        this.scene.camera.setPatternGenerator(this.sampleGenerator, new float2(Math.fround(1 / w), Math.fround(1 / h)));

        // Depth-only consumers still need the geometry pass: draw posW into a scratch target.
        let targets: { channel: Channel; texture: Texture }[] = present.map((ch) => ({ channel: ch, texture: renderData.getTexture(ch.name)! }));
        if (targets.length === 0) {
            if (!this.scratchTarget || this.scratchTarget.width !== w || this.scratchTarget.height !== h) {
                this.scratchTarget = new Texture(this.device, {
                    type: ResourceType.Texture2D,
                    width: w,
                    height: h,
                    format: ResourceFormat.RGBA32Float,
                    bindFlags: ResourceBindFlags.RenderTarget,
                    name: "GBufferRaster::scratch",
                });
            }
            targets = [{ channel: kChannels[0]!, texture: this.scratchTarget }];
        }

        // The vbuffer channel needs the triangle index: pull vertices (non-indexed) for every batch.
        const pulling = targets.some((t) => t.channel.name === "vbuffer");

        // One geometry pass per batch of <= 8 targets. Each batch re-resolves
        // visibility itself (depth cleared, Less + write): identical draw order and
        // vertex math give identical depth, so the batches agree per pixel.
        for (let b = 0; b < targets.length; b += kMaxTargetsPerBatch) {
            const batch = targets.slice(b, b + kMaxTargetsPerBatch);
            const variant = this.getVariant(batch.map((t) => t.channel), pulling);

            const fbo = new Fbo();
            batch.forEach((t, i) => fbo.attachColorTarget(t.texture, i));
            fbo.attachDepthStencilTarget(depth);
            if (b > 0) ctx.clearDsv(depth.getDSV(), 1, 0);

            this.scene.bindShaderData(variant.root);
            variant.root["PerFrameCB"]["gFrameDim"] = [w, h];

            const state = variant.state;
            state.setFbo(fbo);
            const gso = state.getGSO(variant.pipelineLayout);
            const vao = state.getVao()!;
            const bindGroups = variant.vars.getGroupIndices().map((g) => ({ index: g, group: variant.vars.getBindGroup(g) }));

            const desc = fbo.getGpuRenderPassDescriptor();
            const tw = this.device.profilerHook?.passTimestampWrites();
            if (tw) desc.timestampWrites = tw;
            const pass = ctx.getEncoder().beginRenderPass(desc);
            pass.setPipeline(gso.gpuPipeline);
            pass.setViewport(0, 0, fbo.width, fbo.height, 0, 1);
            for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
            vao.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
            if (pulling) {
                // Non-indexed: vertexID counts 0..3*triCount per draw (vertex pulling).
                for (const draw of this.scene.getMeshDrawData().draws) pass.draw(draw.indexCount, 1, 0, draw.firstInstance);
            } else {
                pass.setIndexBuffer(vao.indexBuffer!.gpuBuffer, vao.getGpuIndexFormat());
                for (const draw of this.scene.getMeshDrawData().draws) {
                    pass.drawIndexed(draw.indexCount, 1, draw.firstIndex, draw.baseVertex, draw.firstInstance);
                }
            }
            pass.end();
        }
    }
}

registerRenderPass("GBufferRaster", (device, props) => new GBufferRaster(device, props));
