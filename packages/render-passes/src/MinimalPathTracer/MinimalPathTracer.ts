/**
 * Minimal path tracer mirroring Source/RenderPasses/MinimalPathTracer.
 * Shader is the WebFalcor megakernel override (DESIGN.md §5): the upstream
 * path-tracing logic runs as a compute kernel over software ray queries.
 */

import {
    ComputePass,
    FieldFlags,
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
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/MinimalPathTracer/MinimalPathTracer.rt.slang";

export class MinimalPathTracer extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private maxBounces = 3;

    constructor(device: Device, props: Properties) {
        super(device);
        this.maxBounces = props.get("maxBounces", 3);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Packed hit information").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("viewW", "View direction (world)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("color", "Output radiance")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.pass = null;
        this.frameCount = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const color = renderData.getTexture("color")!;
        if (!this.pass) {
            const defines = this.scene.getSceneDefines().addAll({
                MAX_BOUNCES: this.maxBounces,
                COMPUTE_DIRECT: 1,
                USE_IMPORTANCE_SAMPLING: 1,
                USE_ANALYTIC_LIGHTS: 1,
                USE_EMISSIVE_LIGHTS: 0,
                USE_ENV_LIGHT: 0,
                USE_ENV_BACKGROUND: 0,
                is_valid_gViewW: 0,
            });
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
        }
        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        root["CB"]["gFrameDim"] = [color.width, color.height];
        root["CB"]["gFrameCount"] = this.frameCount;
        root["CB"]["gPRNGDimension"] = 0;
        root["gVBuffer"] = renderData.getTexture("vbuffer")!;
        root["gOutputColor"] = color;
        this.pass.execute(ctx, color.width, color.height);
        this.frameCount++;
    }
}

registerRenderPass("MinimalPathTracer", (device, props) => new MinimalPathTracer(device, props));
