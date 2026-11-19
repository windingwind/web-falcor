/**
 * StyleAovs: object/material IDs and stylization helper buffers decoded from the V-buffer.
 *
 * Stylized renderers key many effects on per-object data that a G-buffer does not carry directly: ID-based outlines,
 * per-object flat colours, colour masks, half-Lambert ramps. This pass decodes the V-buffer hit (instance -> geometry
 * instance data -> material) and writes:
 *  - id         : hash01 of the selected id (`idSource` = instance | material | mesh) in r, the raw id in g;
 *  - idColor    : a random flat colour per id (three independent hashes);
 *  - materialId : hash01 of the material id in r, the raw id in g;
 *  - flatAlbedo : the material base colour sampled at a coarse LOD (`flatLod`; the texture's mean colour when the texture
 *                 has a full mip chain), i.e. "flat" albedo without texture detail;
 *  - ndotl      : unclamped N.L of the key light in r (half-Lambert 0.5 N.L + 0.5 in g); needs the `normW` input;
 *  - ids        : all raw ids at once: geometry instance id (r), material id (g), mesh id (b);
 *  - posObj     : object-space position (the triangle's object-space vertices interpolated with the hit barycentrics);
 *  - curvature  : signed raw curvature in 1/world units in r (> 0 convex, < 0 concave), estimated from the posW/normW
 *                 4-neighbours as the mean of dot(n_j - n, x_j - x) / |x_j - x|^2; neighbours across a depth jump larger
 *                 than `curvatureDepthThreshold` x view depth (or without geometry) are skipped. Needs `posW` and `normW`.
 *                 The raw value only: no thresholding, remapping or shading;
 *  - custom     : a per-instance scalar AOV: r = `instanceScalars[instance id]` (0 for instances past the end of the
 *                 list), a = 1. The scalars come from the `instanceScalars` property or `setInstanceScalars()`; they let
 *                 later passes read any per-object value (a mask weight, a stylization strength, ...).
 * Background pixels are 0 (alpha 0). All outputs are optional and only written when connected/marked. The `custom`
 * output is written by a separate small kernel (StyleAovsCustom.cs.slang) that runs only while `custom` is used, so the
 * main kernel and its outputs are unchanged and it stays within WebGPU's 8 storage textures per shader stage.
 *
 * Inputs : vbuffer (GBufferRaster/VBufferRaster), texC (optional, for flatAlbedo), normW (optional, for ndotl and
 *          curvature), posW (optional, for curvature).
 * Params : idSource {instance|material|mesh}, hashSeed (uint), flatLod (default 16),
 *          lightDir ([0,0,0] = first directional light of the scene, else the given direction towards the light),
 *          curvatureDepthThreshold (relative view-depth jump that separates curvature neighbours, default 0.05).
 * Hash: PCG (Jarzynski & Olano 2020), written for this pass.
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
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, cameraData, dummyTexture, enumIndex, keyLightDir } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/StyleAovs.cs.slang";
const kCustomShaderFile = "RenderPasses/Stylize/StyleAovsCustom.cs.slang";
/** Ordered option names; preserve this order for numeric property compatibility. */
const kIdSources = ["instance", "material", "mesh"] as const;
const kOutputs = [
    ["id", "gId", "hash01(id) in r, raw id in g (id source: idSource)"],
    ["idColor", "gIdColor", "Random flat colour per id"],
    ["materialId", "gMaterialId", "hash01(material id) in r, raw material id in g"],
    ["flatAlbedo", "gFlatAlbedo", "Material base colour at a coarse texture LOD"],
    ["ndotl", "gNdotL", "Unclamped N.L of the key light (r), half-Lambert (g)"],
    ["ids", "gIds", "Raw instance (r), material (g) and mesh (b) ids"],
    ["posObj", "gPosObj", "Object-space position"],
    ["curvature", "gCurvature", "Signed raw curvature in 1/world units (r)"],
] as const;
const kCustomOutput = "custom";

/** Produces geometry, identity and lighting inputs for the stylization passes. */
export class StyleAovs extends RenderPass {
    private mParameters: { idSource: string; hashSeed: number; flatLod: number; lightDir: number[]; curvatureDepthThreshold: number } = {
        idSource: "instance" as string,
        hashSeed: 0,
        flatLod: 16,
        lightDir: [0, 0, 0] as number[],
        curvatureDepthThreshold: 0.05,
    };
    private mPass: ComputePass | null = null;
    private mPassKey: string = "";
    private mCustomPass: ComputePass | null = null;
    private mCustomPassKey: string = "";
    /** Per-instance scalars for the `custom` output (indexed by geometry instance id). */
    private mInstanceScalars: number[] = [];
    private mScalarBuffer: Buffer | null = null;
    private mScalarsDirty: boolean = true;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.mParameters) as (keyof typeof this.mParameters)[])
            (this.mParameters as Record<string, unknown>)[k] = props.get(k, this.mParameters[k]);
        enumIndex("StyleAovs", "idSource", this.mParameters.idSource, kIdSources);
        this.setInstanceScalars(props.get<number[]>("instanceScalars", []));
    }

    override getProperties(): Properties {
        const p = new Properties({ ...this.mParameters, lightDir: [...this.mParameters.lightDir] });
        if (this.mInstanceScalars.length > 0) p.set("instanceScalars", [...this.mInstanceScalars]);
        return p;
    }

    /** Sets the per-instance scalars read by the `custom` output (index = geometry instance id). */
    setInstanceScalars(values: ArrayLike<number>): void {
        this.mInstanceScalars = Array.from(values, (v) => Number(v));
        this.mScalarsDirty = true;
    }

    getInstanceScalars(): number[] {
        return [...this.mInstanceScalars];
    }

    private updateScalarBuffer(): Buffer {
        const n = Math.max(1, this.mInstanceScalars.length);
        if (!this.mScalarBuffer || this.mScalarBuffer.size < n * 4) {
            this.mScalarBuffer = new Buffer(this.device, {
                size: n * 4,
                structSize: 4,
                bindFlags: ResourceBindFlags.ShaderResource,
                memoryType: MemoryType.DeviceLocal,
                name: "StyleAovs::instanceScalars",
            });
            this.mScalarsDirty = true;
        }
        if (this.mScalarsDirty) {
            const data = new Float32Array(n);
            data.set(this.mInstanceScalars);
            this.mScalarBuffer.setBlob(new Uint8Array(data.buffer));
            this.mScalarsDirty = false;
        }
        return this.mScalarBuffer;
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Hash seed", this.mParameters.hashSeed, 0, 1000, 1, (v) => (this.mParameters.hashSeed = v));
        ui.slider("Flat albedo LOD", this.mParameters.flatLod, 0, 16, 0.5, (v) => (this.mParameters.flatLod = v));
        ui.slider(
            "Curvature depth threshold",
            this.mParameters.curvatureDepthThreshold,
            0.001,
            1,
            0.001,
            (v) => (this.mParameters.curvatureDepthThreshold = v),
        );
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Visibility buffer").bindFlags(ResourceBindFlags.ShaderResource);
        addTexInput(r, "texC", "Texture coordinates (for flatAlbedo)", true);
        addTexInput(r, "normW", "World-space shading normal (for ndotl and curvature)", true);
        addTexInput(r, "posW", "World-space position (for curvature)", true);
        for (const [name, , desc] of kOutputs) addColorOutput(r, name, desc, w, h, true);
        addColorOutput(r, kCustomOutput, "Per-instance scalar AOV: instanceScalars[instance id] in r", w, h, true);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const vbuffer = rd.getTexture("vbuffer");
        if (!vbuffer) return;
        if (!this.scene) throw new RuntimeError("StyleAovs: requires a scene");
        const W = vbuffer.width;
        const H = vbuffer.height;
        const defines = this.scene.getSceneDefines();
        const custom = rd.getTexture(kCustomOutput);
        const key = defines.key();
        if (!this.mPass || key !== this.mPassKey) {
            this.mPass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.mPassKey = key;
        }
        const root = this.mPass.getRootVar();
        this.scene.bindShaderData(root);
        const texC = rd.getTexture("texC");
        const normW = rd.getTexture("normW");
        const posW = rd.getTexture("posW");
        const L = keyLightDir(this.scene, this.mParameters.lightDir);
        const cb = root["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gIdSource"] = enumIndex("StyleAovs", "idSource", this.mParameters.idSource, kIdSources);
        cb["gSeed"] = Math.max(0, Math.round(this.mParameters.hashSeed)) >>> 0;
        cb["gLightDir"] = L;
        cb["gFlatLod"] = this.mParameters.flatLod;
        cb["gHasTexC"] = texC ? 1 : 0;
        cb["gHasNormW"] = normW ? 1 : 0;
        cb["gHasPosW"] = posW ? 1 : 0;
        cb["gCurvDepthThreshold"] = this.mParameters.curvatureDepthThreshold;
        const cam = cameraData(this.scene);
        cb["gCamPos"] = cam.pos;
        cb["gCamFwd"] = cam.fwd;
        const dummy = dummyTexture(this.device);
        root["gVBuffer"] = vbuffer;
        root["gTexC"] = texC ?? dummy;
        root["gNormW"] = normW ?? dummy;
        root["gPosW"] = posW ?? dummy;
        let any = false;
        for (const [name, tex] of kOutputs) {
            const t = rd.getTexture(name);
            root[tex] = t ?? dummyTexture(this.device, `out:${name}`);
            any ||= !!t;
        }
        if (any) this.mPass.execute(ctx, W, H);
        if (custom) {
            if (!this.mCustomPass || key !== this.mCustomPassKey) {
                this.mCustomPass = ComputePass.create(this.device, { path: kCustomShaderFile, defines });
                this.mCustomPassKey = key;
            }
            const croot = this.mCustomPass.getRootVar();
            const ccb = croot["CB"]!;
            ccb["gFrameDim"] = [W, H];
            ccb["gInstanceScalarCount"] = this.mInstanceScalars.length;
            croot["gVBuffer"] = vbuffer;
            croot["gInstanceScalars"] = this.updateScalarBuffer();
            croot["gCustom"] = custom;
            this.mCustomPass.execute(ctx, W, H);
        }
    }
}

registerRenderPass("StyleAovs", (device, props) => new StyleAovs(device, props), {
    desc: "Object/material IDs, ID colours, flat albedo, N.L, object-space position and raw curvature, from the V-buffer",
});
