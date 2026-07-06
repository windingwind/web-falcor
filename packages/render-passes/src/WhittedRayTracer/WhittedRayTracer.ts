/**
 * Whitted ray tracer mirroring Source/RenderPasses/WhittedRayTracer, running
 * as the WebFalcor compute megakernel over software ray queries (see the
 * WhittedRayTracer.rt.slang override). Consumes the G-buffer produced by
 * GBufferRT and follows perfect reflections/refractions with texture LOD via
 * ray cones or ray differentials.
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
    SAMPLE_GENERATOR_DEFAULT,
    SampleGenerator,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/WhittedRayTracer/WhittedRayTracer.rt.slang";

const kTexLODModes: Record<string, number> = { Mip0: 0, RayCones: 1, RayDiffs: 2 };
const kRayConeModes: Record<string, number> = { Combo: 0, Unified: 1 };
const kFilterModes: Record<string, number> = { Isotropic: 0, Anisotropic: 1, AnisotropicWhenRefraction: 2 };

/** [input name, shader texture name] (kInputChannels). */
const kInputs: [string, string][] = [
    ["posW", "gWorldPosition"],
    ["normalW", "gWorldShadingNormal"],
    ["tangentW", "gWorldShadingTangent"],
    ["faceNormalW", "gWorldFaceNormal"],
    ["texC", "gTextureCoord"],
    ["texGrads", "gTextureGrads"],
    ["mtlData", "gMaterialData"],
    ["vbuffer", "gVBuffer"],
];

export class WhittedRayTracer extends RenderPass {
    private pass: ComputePass | null = null;
    private passKey = "";
    private frameCount = 0;
    private maxBounces = 3;
    private texLODMode = 0;
    private rayConeMode = 1;
    private rayConeFilterMode = 0;
    private rayDiffFilterMode = 0;
    private useRoughnessToVariance = false;
    private sampleGenerator: SampleGenerator;

    constructor(device: Device, props: Properties) {
        super(device);
        this.maxBounces = props.get("maxBounces", 3);
        const parse = (name: string, table: Record<string, number>, fallback: number) => {
            const v = props.getOpt<string | number>(name);
            if (v === undefined) return fallback;
            return (typeof v === "string" ? table[v] : v) ?? fallback;
        };
        this.texLODMode = parse("texLODMode", kTexLODModes, 0);
        this.rayConeMode = parse("rayConeMode", kRayConeModes, 1);
        this.rayConeFilterMode = parse("rayConeFilterMode", kFilterModes, 0);
        this.rayDiffFilterMode = parse("rayDiffFilterMode", kFilterModes, 0);
        this.useRoughnessToVariance = props.get("useRoughnessToVariance", false);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_DEFAULT);
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        for (const [name] of kInputs) {
            const f = r.addInput(name, name).bindFlags(ResourceBindFlags.ShaderResource);
            if (name === "texGrads" || name === "vbuffer") f.flags(FieldFlags.Optional);
        }
        r.addOutput("color", "Output color (sum of direct and indirect)")
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
        const [w, h] = [color.width, color.height];

        const valid: Record<string, number> = {};
        for (const [name, texname] of kInputs) valid[`is_valid_${texname}`] = renderData.getTexture(name) ? 1 : 0;
        const key = JSON.stringify(valid);
        if (!this.pass || this.passKey !== key) {
            const defines = this.scene.getSceneDefines().addAll({
                MAX_BOUNCES: this.maxBounces,
                TEX_LOD_MODE: this.texLODMode,
                RAY_CONE_MODE: this.rayConeMode,
                RAY_CONE_FILTER_MODE: this.rayConeFilterMode,
                RAY_DIFF_FILTER_MODE: this.rayDiffFilterMode,
                USE_ROUGHNESS_TO_VARIANCE: this.useRoughnessToVariance ? 1 : 0,
                VISUALIZE_SURFACE_SPREAD: 0,
                USE_FRESNEL_AS_BRDF: 0,
                USE_ANALYTIC_LIGHTS: this.scene.useAnalyticLights ? 1 : 0,
                USE_EMISSIVE_LIGHTS: this.scene.useEmissiveLights ? 1 : 0,
                USE_ENV_LIGHT: this.scene.useEnvLight ? 1 : 0,
                USE_ENV_BACKGROUND: this.scene.useEnvBackground ? 1 : 0,
                ...valid,
            });
            defines.addAll(this.sampleGenerator.getDefines());
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.passKey = key;
        }

        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        root["CB"]["gFrameCount"] = this.frameCount;
        root["CB"]["gPRNGDimension"] = 0;
        // Mirrors Camera::computeScreenSpacePixelSpreadAngle (kDefaultFrameHeight = 24).
        const fovY = 2 * Math.atan(0.5 * 24 / this.scene.camera.getFocalLength());
        root["CB"]["gScreenSpacePixelSpreadAngle"] = Math.fround(Math.atan((2 * Math.tan(fovY * 0.5)) / h));
        root["CB"]["gFrameDim"] = [w, h];
        for (const [name, texname] of kInputs) {
            const tex = renderData.getTexture(name);
            if (tex) root[texname] = tex;
        }
        root["gOutputColor"] = color;
        this.pass.execute(ctx, w, h);
        this.frameCount++;
    }
}

registerRenderPass("WhittedRayTracer", (device, props) => new WhittedRayTracer(device, props));
