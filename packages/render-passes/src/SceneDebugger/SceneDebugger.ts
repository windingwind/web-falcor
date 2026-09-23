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
    PixelDebug,
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
    type UIWidgets,
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

/** Mirrors getModeDesc. */
const kModeDesc: Record<string, string> = {
    FlatShaded: "Flat shaded",
    TriangleDensity: "Triangle density",
    HitType: "Hit type in pseudocolor",
    InstanceID: "Instance ID in pseudocolor",
    MaterialID: "Material ID in pseudocolor",
    PrimitiveID: "Primitive ID in pseudocolor",
    GeometryID: "Geometry ID in pseudocolor",
    BlasID: "Raytracing bottom-level acceleration structure (BLAS) ID in pseudocolor",
    InstancedGeometry: "Green = instanced geometry, red = non-instanced geometry",
    MaterialType: "Material type in pseudocolor",
    FaceNormal: "Face normal in RGB color",
    ShadingNormal: "Shading normal in RGB color",
    ShadingTangent: "Shading tangent in RGB color",
    ShadingBitangent: "Shading bitangent in RGB color",
    FrontFacingFlag: "Green = front-facing, red = back-facing",
    BackfacingShadingNormal: "Pixels where the shading normal is back-facing with respect to view vector are highlighted",
    TexCoords: "Texture coordinates in RG color wrapped to [0,1]",
    BSDFProperties: "BSDF properties",
};

export class SceneDebugger extends RenderPass {
    private pass: ComputePass | null = null;
    private frameCount = 0;
    /** Mirrors mpPixelDebug (UI under "Debugging"; left click selects the pixel). */
    readonly pixelDebug = new PixelDebug(this.device);
    private frameDim: [number, number] = [0, 0];
    private modeValue = kModes["FaceNormal"]!;

    /** Mirrors the Python `mode` property (the SceneDebuggerMode enum name). */
    get mode(): string {
        return Object.keys(kModes).find((k) => kModes[k] === this.modeValue) ?? "FaceNormal";
    }
    set mode(value: string) {
        const mode = kModes[value];
        if (mode === undefined) throw new Error(`Invalid SceneDebuggerMode '${value}'`);
        this.modeValue = mode;
    }
    private bsdfProperty = 0;
    // Remaining SceneDebuggerParams (native defaults).
    private bsdfIndex = 0;
    private clamp = true;
    private flipSign = false;
    private remapRange = true;
    private showVolumes = true;
    private volumeDensityScale = 1;
    private triangleDensityLogRange: [number, number] = [-16, 16];
    private selectedPixel: [number, number] = [0, 0];
    private pixelData: Buffer | null = null;
    private dummyVbuffer: Texture | null = null;
    private meshToBlasID: Buffer | null = null;
    private instanceInfo: Buffer | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        const mode = props.getOpt<string | number>("mode");
        if (mode !== undefined) this.modeValue = (typeof mode === "string" ? kModes[mode] : mode) ?? this.modeValue;
        // Which BSDF property BSDFProperties mode visualizes (SceneDebuggerBSDFProperty:
        // 0 Emission, 1 Roughness, 2 GuideNormal, 3 DiffuseReflectionAlbedo, ...).
        const bp = props.getOpt<string | number>("bsdfProperty");
        if (bp !== undefined) this.bsdfProperty = (typeof bp === "string" ? kBSDFProps[bp] : bp) ?? this.bsdfProperty;
    }

    override getProperties(): Properties {
        const name = (table: Record<string, number>, v: number) => Object.keys(table).find((k) => table[k] === v) ?? v;
        return new Properties({ mode: name(kModes, this.modeValue), bsdfProperty: name(kBSDFProps, this.bsdfProperty) });
    }

    /** Mirrors SceneDebugger::renderUI (all runtime parameters; pixel-data readout lives in the viewer's picking). */
    override renderUI(ui: UIWidgets): void {
        const name = (table: Record<string, number>, v: number) => Object.keys(table).find((k) => table[k] === v) ?? Object.keys(table)[0]!;
        ui.dropdown("Mode", Object.keys(kModes), name(kModes, this.modeValue), (v) => (this.modeValue = kModes[v]!));
        ui.slider("Triangle density range min (log2)", this.triangleDensityLogRange[0], -32, 32, 1, (v) => (this.triangleDensityLogRange = [Math.round(v), this.triangleDensityLogRange[1]]));
        ui.slider("Triangle density range max (log2)", this.triangleDensityLogRange[1], -32, 32, 1, (v) => (this.triangleDensityLogRange = [this.triangleDensityLogRange[0], Math.round(v)]));
        ui.dropdown("BSDF property", Object.keys(kBSDFProps), name(kBSDFProps, this.bsdfProperty), (v) => (this.bsdfProperty = kBSDFProps[v]!));
        ui.slider("BSDF index", this.bsdfIndex, 0, 15, 1, (v) => (this.bsdfIndex = Math.round(v)));
        ui.checkbox("Clamp to [0,1]", this.clamp, (v) => (this.clamp = v));
        ui.checkbox("Flip sign", this.flipSign, (v) => (this.flipSign = v));
        ui.checkbox("Remap to [0,1]", this.remapRange, (v) => (this.remapRange = v));
        ui.checkbox("Show volumes", this.showVolumes, (v) => (this.showVolumes = v));
        ui.slider("Volume density scale", this.volumeDensityScale, 0, 1000, 0.1, (v) => (this.volumeDensityScale = v));
        ui.text(`Description: ${kModeDesc[name(kModes, this.modeValue)] ?? ""}`);
        this.pixelDebug.renderUI(ui.group("Debugging"), () => (this.pass = null));
    }

    /** Mirrors SceneDebugger::onMouseEvent: left click selects the inspected pixel (and the debug pixel). */
    onMouseEvent(ev: { type: "buttonDown" | "buttonUp" | "move"; button?: "left" | "right" | "middle"; pos: [number, number] }): boolean {
        if (ev.type === "buttonDown" && ev.button === "left") {
            const [w, h] = this.frameDim;
            this.selectedPixel = [Math.min(Math.max(Math.trunc(ev.pos[0] * w), 0), Math.max(w - 1, 0)), Math.min(Math.max(Math.trunc(ev.pos[1] * h), 0), Math.max(h - 1, 0))];
        }
        return this.pixelDebug.onMouseEvent(ev);
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
        this.frameDim = [w, h];
        this.pixelDebug.beginFrame(ctx, this.frameDim);

        if (!this.pass) {
            const defines = this.scene.getSceneDefines();
            if (this.pixelDebug.enabled) defines.addAll(PixelDebug.getDefines());
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
        p["mode"] = this.modeValue;
        p["frameDim"] = [w, h];
        p["frameCount"] = this.frameCount;
        p["bsdfProperty"] = this.bsdfProperty;
        p["bsdfIndex"] = this.bsdfIndex;
        p["selectedPixel"] = this.selectedPixel;
        p["flipSign"] = this.flipSign ? 1 : 0;
        p["remapRange"] = this.remapRange ? 1 : 0;
        p["clamp"] = this.clamp ? 1 : 0;
        p["showVolumes"] = this.showVolumes ? 1 : 0;
        p["volumeDensityScale"] = this.volumeDensityScale;
        p["useVBuffer"] = renderData.getTexture("vbuffer") ? 1 : 0;
        p["profileSecondaryRays"] = 0;
        p["profileSecondaryLoadHit"] = 0;
        p["profileSecondaryConeAngle"] = 90;
        p["triangleDensityLogRange"] = this.triangleDensityLogRange;
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
        this.pixelDebug.prepareProgram(root);
        this.pass.execute(ctx, w, h);
        this.pixelDebug.endFrame();
        this.frameCount++;
    }
}

registerRenderPass("SceneDebugger", (device, props) => new SceneDebugger(device, props));
