/**
 * BSDF viewer mirroring Source/RenderPasses/BSDFViewer: renders a unit sphere
 * shaded with one scene material under omnidirectional/env lighting. Material
 * slice-view and pixel readback UI are Mogwai scope; the kernel is unmodified
 * (write-only output override only).
 */

import {
    Buffer,
    ComputePass,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    SAMPLE_GENERATOR_DEFAULT,
    SampleGenerator,
    Sampler,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/BSDFViewer/BSDFViewer.cs.slang";
const kIdentity3x4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

export class BSDFViewer extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private materialID = 0;
    private sampleGenerator: SampleGenerator;
    private pixelData: Buffer | null = null;
    private dummyEnvTex: Texture | null = null;
    private dummySampler: Sampler | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.materialID = props.get("materialID", 0);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_DEFAULT);
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addOutput("output", "Output buffer")
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
        const output = renderData.getTexture("output")!;
        const [w, h] = [output.width, output.height];

        if (!this.pass) {
            const defines = this.scene.getSceneDefines();
            defines.addAll(this.sampleGenerator.getDefines());
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            // Oversized for WGSL std430 (float3 members pad to 16 B).
            this.pixelData = new Buffer(this.device, {
                size: 512,
                structSize: 512,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "BSDFViewer::pixelData",
            });
        }

        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        const v = root["gBSDFViewer"] as ShaderVar;
        const p = v["params"] as ShaderVar;
        p["frameDim"] = [w, h];
        p["frameCount"] = this.frameCount;
        // Mirrors BSDFViewer::compile: centered square viewport.
        const extent = Math.min(w, h);
        p["viewportOffset"] = [Math.floor((w - extent) / 2), Math.floor((h - extent) / 2)];
        p["viewportScale"] = [Math.fround(1 / extent), Math.fround(1 / extent)];
        p["materialID"] = this.materialID;
        p["useNormalMapping"] = 0;
        p["useFixedTexCoords"] = 0;
        p["texCoords"] = [0, 0];
        p["useDisneyDiffuse"] = 0;
        p["useSeparableMaskingShadowing"] = 0;
        p["useImportanceSampling"] = 1;
        p["usePdf"] = 0;
        p["outputAlbedo"] = 0;
        p["enableDiffuse"] = 1;
        p["enableSpecular"] = 1;
        p["applyNdotL"] = 0;
        p["useGroundPlane"] = 0;
        p["useEnvMap"] = this.scene.useEnvLight ? 1 : 0;
        p["lightIntensity"] = 1;
        p["lightColor"] = [1, 1, 1];
        p["useDirectionalLight"] = 0;
        p["lightDir"] = [0, 0, -1];
        p["orthographicCamera"] = 0;
        const cameraDistance = 1.5;
        p["cameraDistance"] = cameraDistance;
        p["cameraFovY"] = 90;
        // Mirrors native runtime computation: tan(fovY/2) * distance.
        p["cameraViewportScale"] = Math.fround(Math.tan((90 / 2) * (Math.PI / 180)) * cameraDistance);
        p["selectedPixel"] = [0, 0];

        // EnvMap struct member survives DCE; bind the scene envmap or dummies.
        const env = this.scene.getEnvMap();
        const envVar = v["envMap"] as ShaderVar;
        if (env) {
            env.bindShaderData(envVar);
        } else {
            this.dummyEnvTex ??= new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: 1,
                height: 1,
                format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.ShaderResource,
                name: "BSDFViewer::dummyEnv",
            });
            this.dummySampler ??= new Sampler(this.device, {});
            const data = envVar["data"] as ShaderVar;
            data["transform"] = kIdentity3x4;
            data["invTransform"] = kIdentity3x4;
            data["tint"] = [1, 1, 1];
            data["intensity"] = 1;
            envVar["envMap"] = this.dummyEnvTex;
            envVar["envSampler"] = this.dummySampler;
        }

        v["outputColor"] = output;
        v["pixelData"] = this.pixelData!;
        this.pass.execute(ctx, w, h);
        this.frameCount++;
    }
}

registerRenderPass("BSDFViewer", (device, props) => new BSDFViewer(device, props));
