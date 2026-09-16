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
    type UIWidgets,
    MaterialType,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/BSDFViewer/BSDFViewer.cs.slang";
const kIdentity3x4 = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0];

export class BSDFViewer extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private materialID = 0;
    // BSDFViewerParams (native defaults); Material viewer mode only on the web.
    private useNormalMapping = false;
    private useFixedTexCoords = false;
    private texCoords: [number, number] = [0, 0];
    private useDisneyDiffuse = false;
    private useSeparableMaskingShadowing = false;
    private useImportanceSampling = true;
    private usePdf = false;
    private outputAlbedo = 0;
    private enableDiffuse = true;
    private enableSpecular = true;
    private applyNdotL = false;
    private useGroundPlane = false;
    private useEnvMap = true;
    private lightIntensity = 1;
    private lightColor: [number, number, number] = [1, 1, 1];
    private useDirectionalLight = false;
    private lightDir: [number, number, number] = [0, 0, -1];
    private orthographicCamera = false;
    private cameraDistance = 1.5;
    private cameraFovY = 90;
    private sampleGenerator: SampleGenerator;
    private pixelData: Buffer | null = null;
    private dummyEnvTex: Texture | null = null;
    private dummySampler: Sampler | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.materialID = props.get("materialID", 0);
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_DEFAULT);
    }

    override getProperties(): Properties {
        return new Properties({ materialID: this.materialID });
    }

    /** Mirrors BSDFViewer::renderUI (Material viewer mode; every change restarts accumulation). */
    override renderUI(ui: UIWidgets): void {
        if (!this.scene || this.scene.getMaterialCount() === 0) {
            ui.text("No scene/materials loaded");
            return;
        }
        const dirty = <T>(set: (v: T) => void) => (v: T) => {
            set(v);
            this.frameCount = 0;
        };
        ui.text("The current mode shows a shaded unit sphere. The coordinate frame is right-handed with xy pointing right/up and +z towards the viewer.");
        const mtl = ui.group("Material");
        const names: string[] = [];
        for (let i = 0; i < this.scene.getMaterialCount(); i++) names.push(`${i}: ${this.scene.getMaterial(i).name}`);
        mtl.dropdown("Materials", names, names[Math.min(this.materialID, names.length - 1)]!, dirty((v: string) => (this.materialID = Number(v.split(":")[0]))));
        const type = this.scene.getMaterial(Math.min(this.materialID, names.length - 1)).header?.materialType;
        mtl.text(`Material type: ${type !== undefined ? MaterialType[type] : "?"}`);
        mtl.checkbox("Use normal mapping", this.useNormalMapping, dirty((v) => (this.useNormalMapping = v)));
        mtl.checkbox("Fixed tex coords", this.useFixedTexCoords, dirty((v) => (this.useFixedTexCoords = v)));
        mtl.slider("Tex coord U", this.texCoords[0], 0, 1, 0.01, dirty((v) => (this.texCoords = [v, this.texCoords[1]])));
        mtl.slider("Tex coord V", this.texCoords[1], 0, 1, 0.01, dirty((v) => (this.texCoords = [this.texCoords[0], v])));
        const bsdf = ui.group("BSDF");
        bsdf.checkbox("Enable diffuse", this.enableDiffuse, dirty((v) => (this.enableDiffuse = v)));
        bsdf.checkbox("Enable specular", this.enableSpecular, dirty((v) => (this.enableSpecular = v)));
        bsdf.checkbox("Use Disney' diffuse BRDF", this.useDisneyDiffuse, dirty((v) => (this.useDisneyDiffuse = v)));
        bsdf.checkbox("Use separable masking-shadowing", this.useSeparableMaskingShadowing, dirty((v) => (this.useSeparableMaskingShadowing = v)));
        bsdf.checkbox("Use importance sampling", this.useImportanceSampling, dirty((v) => (this.useImportanceSampling = v)));
        bsdf.checkbox("Use pdf", this.usePdf, dirty((v) => (this.usePdf = v)));
        bsdf.text("Material viewer settings:");
        // AlbedoSelection flags: ShowAlbedo 1, DiffuseReflection 2, DiffuseTransmission 4, SpecularReflection 8, SpecularTransmission 16.
        const flag = (label: string, bit: number) => bsdf.checkbox(label, (this.outputAlbedo & bit) !== 0, dirty((v: boolean) => (this.outputAlbedo = v ? this.outputAlbedo | bit : this.outputAlbedo & ~bit)));
        flag("Show albedo", 1);
        flag("Diffuse reflection", 2);
        flag("Diffuse transmission", 4);
        flag("Specular reflection", 8);
        flag("Specular transmission", 16);
        bsdf.checkbox("Apply NdotL", this.applyNdotL, dirty((v) => (this.applyNdotL = v)));
        const light = ui.group("Light");
        light.checkbox("Use env map", this.useEnvMap, dirty((v) => (this.useEnvMap = v)));
        light.checkbox("Use directional light", this.useDirectionalLight, dirty((v) => (this.useDirectionalLight = v)));
        light.slider("Light intensity", this.lightIntensity, 0, 10, 0.01, dirty((v) => (this.lightIntensity = v)));
        ["R", "G", "B"].forEach((c, i) => light.slider(`Light color ${c}`, this.lightColor[i]!, 0, 1, 0.01, dirty((v: number) => (this.lightColor[i] = v))));
        ["X", "Y", "Z"].forEach((c, i) => light.slider(`Light dir ${c}`, this.lightDir[i]!, -1, 1, 0.01, dirty((v: number) => (this.lightDir[i] = v))));
        light.checkbox("Ground plane", this.useGroundPlane, dirty((v) => (this.useGroundPlane = v)));
        const cam = ui.group("Camera");
        cam.checkbox("Orthographic", this.orthographicCamera, dirty((v) => (this.orthographicCamera = v)));
        cam.slider("Distance", this.cameraDistance, 1.01, 10, 0.01, dirty((v) => (this.cameraDistance = v)));
        cam.slider("FOV (y)", this.cameraFovY, 1, 179, 1, dirty((v) => (this.cameraFovY = v)));
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
        p["materialID"] = Math.min(this.materialID, Math.max(0, this.scene.getMaterialCount() - 1));
        p["useNormalMapping"] = this.useNormalMapping ? 1 : 0;
        p["useFixedTexCoords"] = this.useFixedTexCoords ? 1 : 0;
        p["texCoords"] = this.texCoords;
        p["useDisneyDiffuse"] = this.useDisneyDiffuse ? 1 : 0;
        p["useSeparableMaskingShadowing"] = this.useSeparableMaskingShadowing ? 1 : 0;
        p["useImportanceSampling"] = this.useImportanceSampling ? 1 : 0;
        p["usePdf"] = this.usePdf ? 1 : 0;
        p["outputAlbedo"] = this.outputAlbedo;
        p["enableDiffuse"] = this.enableDiffuse ? 1 : 0;
        p["enableSpecular"] = this.enableSpecular ? 1 : 0;
        p["applyNdotL"] = this.applyNdotL ? 1 : 0;
        p["useGroundPlane"] = this.useGroundPlane ? 1 : 0;
        p["useEnvMap"] = this.useEnvMap && this.scene.useEnvLight ? 1 : 0;
        p["lightIntensity"] = this.lightIntensity;
        p["lightColor"] = this.lightColor;
        p["useDirectionalLight"] = this.useDirectionalLight ? 1 : 0;
        p["lightDir"] = this.lightDir;
        p["orthographicCamera"] = this.orthographicCamera ? 1 : 0;
        p["cameraDistance"] = this.cameraDistance;
        p["cameraFovY"] = this.cameraFovY;
        // Mirrors native runtime computation: tan(fovY/2) * distance.
        p["cameraViewportScale"] = Math.fround(Math.tan((this.cameraFovY / 2) * (Math.PI / 180)) * this.cameraDistance);
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
