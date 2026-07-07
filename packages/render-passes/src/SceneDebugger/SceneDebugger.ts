/**
 * Scene debugger mirroring Source/RenderPasses/SceneDebugger: visualizes
 * geometry/shading attributes by tracing primary rays (inline queries).
 * Mouse picking (pixelData readback UI) is Mogwai-UI scope; the readback
 * buffer is still bound so the kernel is unmodified.
 *
 * Web note: instances are baked one-mesh-per-instance by the SceneBuilder, so
 * meshToBlasID is the identity and no geometry is flagged IsInstanced.
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type ShaderVar,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/SceneDebugger/SceneDebugger.cs.slang";

/** Mirrors SceneDebuggerMode (SharedTypes.slang). */
const kModes: Record<string, number> = {
    FlatShaded: 0, TriangleDensity: 1, HitType: 2, InstanceID: 3, MaterialID: 4, GeometryID: 5,
    BlasID: 6, PrimitiveID: 7, InstancedGeometry: 8, MaterialType: 9, FaceNormal: 10,
    ShadingNormal: 11, ShadingTangent: 12, ShadingBitangent: 13, FrontFacingFlag: 14,
    BackfacingShadingNormal: 15, TexCoords: 16, BSDFProperties: 17,
};

/** Mirrors SceneDebuggerBSDFProperty (SharedTypes.slang). */
const kBSDFProps: Record<string, number> = {
    Emission: 0, Roughness: 1, GuideNormal: 2, DiffuseReflectionAlbedo: 3, DiffuseTransmissionAlbedo: 4,
    SpecularReflectionAlbedo: 5, SpecularTransmissionAlbedo: 6, SpecularReflectance: 7, IsTransmissive: 8,
};

export class SceneDebugger extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    private mode = kModes["FaceNormal"]!;
    private bsdfProperty = 0;
    private pixelData: Buffer | null = null;
    private dummyVbuffer: Texture | null = null;
    private meshToBlasID: Buffer | null = null;
    private instanceInfo: Buffer | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        const mode = props.getOpt<string | number>("mode");
        if (mode !== undefined) this.mode = (typeof mode === "string" ? kModes[mode] : mode) ?? this.mode;
        // Which BSDF property BSDFProperties mode visualizes (SceneDebuggerBSDFProperty:
        // 0 Emission, 1 Roughness, 2 GuideNormal, 3 DiffuseReflectionAlbedo, ...).
        const bp = props.getOpt<string | number>("bsdfProperty");
        if (bp !== undefined) this.bsdfProperty = (typeof bp === "string" ? kBSDFProps[bp] : bp) ?? this.bsdfProperty;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "V-buffer (optional)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("output", "Scene debugger output")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.pass = null;
        this.meshToBlasID = null;
        this.instanceInfo = null;
        this.frameCount = 0;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const output = renderData.getTexture("output")!;
        const [w, h] = [output.width, output.height];

        if (!this.pass) {
            const defines = this.scene.getSceneDefines();
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
        }
        if (!this.meshToBlasID) {
            // Web scenes bake one mesh per instance: identity mapping, no shared BLAS.
            const count = Math.max(1, this.scene.getMeshDrawData().draws.length);
            const ids = new Uint32Array(count);
            for (let i = 0; i < count; i++) ids[i] = i;
            const make = (data: Uint32Array, name: string) => {
                const buf = new Buffer(this.device, {
                    size: data.byteLength,
                    structSize: 4,
                    bindFlags: ResourceBindFlags.ShaderResource,
                    memoryType: MemoryType.DeviceLocal,
                    name,
                });
                buf.setBlob(new Uint8Array(data.buffer));
                return buf;
            };
            this.meshToBlasID = make(ids, "SceneDebugger::meshToBlasID");
            this.instanceInfo = make(new Uint32Array(count), "SceneDebugger::instanceInfo");
            // WGSL std430 PixelData (float3 members pad to 16B): 176 bytes.
            this.pixelData = new Buffer(this.device, {
                size: 176,
                structSize: 176,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "SceneDebugger::pixelData",
            });
        }

        const root = this.pass.getRootVar();
        this.scene.bindShaderData(root);
        const sd = root["CB"]!["gSceneDebugger"] as ShaderVar;
        const p = sd["params"] as ShaderVar;
        p["mode"] = this.mode;
        p["frameDim"] = [w, h];
        p["frameCount"] = this.frameCount;
        p["bsdfProperty"] = this.bsdfProperty;
        p["bsdfIndex"] = 0;
        p["selectedPixel"] = [0, 0];
        p["flipSign"] = 0;
        p["remapRange"] = 1;
        p["clamp"] = 1;
        p["showVolumes"] = 1;
        p["volumeDensityScale"] = 1;
        p["useVBuffer"] = renderData.getTexture("vbuffer") ? 1 : 0;
        p["profileSecondaryRays"] = 0;
        p["profileSecondaryLoadHit"] = 0;
        p["profileSecondaryConeAngle"] = 90;
        p["triangleDensityLogRange"] = [-16, 16];
        sd["meshToBlasID"] = this.meshToBlasID!;
        sd["instanceInfo"] = this.instanceInfo!;
        // vbuffer is gated by a runtime flag, so the binding survives DCE.
        this.dummyVbuffer ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.RGBA32Uint,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "SceneDebugger::dummyVbuffer",
        });
        sd["vbuffer"] = renderData.getTexture("vbuffer") ?? this.dummyVbuffer;
        sd["output"] = output;
        sd["pixelData"] = this.pixelData!;
        this.pass.execute(ctx, w, h);
        this.frameCount++;
    }
}

registerRenderPass("SceneDebugger", (device, props) => new SceneDebugger(device, props));
