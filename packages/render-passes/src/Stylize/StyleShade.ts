/**
 * StyleShade: deferred re-shading of the scene with a per-material style table.
 *
 * Consumes a V-buffer like DeferredLighting and evaluates the same direct-lighting terms (analytic lights, SH9 sky
 * diffuse, env-map specular, emission, env background), but lets every material pick a shading style:
 *  - diffuseModel : lambert | wrap (wrap factor) | halfLambert | minnaert (minnaertK);
 *  - hemisphere ambient : lerp(groundColor, skyColor, 0.5 N.y + 0.5) * albedo, blended over the sky term by hemiMix;
 *  - flatMix      : blends the shaded diffuse towards the plain albedo (unlit look);
 *  - specModel    : ggx (the reference lobe) | blinn (normalized Blinn-Phong, shininess; <= 0 derives it from the
 *                   roughness) | hardSpec (one smooth threshold on the Blinn-Phong lobe: a crisp highlight, the lit and
 *                   shadowed diffuse are not touched);
 *  - specScale    : scales the specular (analytic + env).
 * There is deliberately no colour ramp or band-count mode.
 *
 * The top-level properties form the default record (all materials). `materials` is a list of overrides
 * `{ materialId, ...any record field }`; a material without an override uses the default record. Material ids come
 * from the optional `materialId` input (StyleAovs.materialId: raw id in g) or, when it is not connected, from the
 * V-buffer hit.
 *
 * Shadows are not computed here: connect a visibility texture (r = visibility of one light, e.g.
 * DeferredLighting.visibility or a visibility pass) to `visibility`; it is applied to the light given by `shadowLight`
 * (-1 = the CascadedShadowMap light from the graph dictionary if present, else the first directional light).
 *
 * Identity: styleMix = 0 gives DeferredLighting's LambertGGX colour (same terms, same visibility; equal up to fp32 rounding
 * differences of the separately compiled shaders); so does the
 * default record (lambert, hemiMix 0, flatMix 0, ggx, specScale 1) at any styleMix in {0, 1}.
 *
 * Inputs : vbuffer, viewW (optional), visibility (optional), materialId (optional).
 * Outputs: color, diffuse (optional), specular (optional).
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    LightType,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { kCascadedShadowMapData, type CascadedShadowMapData } from "../RasterLighting/CascadedShadowMap.js";
import { dummyTexture, parseColour } from "./StyleCommon.js";

const kShaderFile = "RenderPasses/Stylize/StyleShade.cs.slang";
const kSHShaderFile = "RenderPasses/RasterLighting/EnvSH.cs.slang";

export const kStyleShadeDiffuseModels = ["lambert", "wrap", "halfLambert", "minnaert"] as const;
export const kStyleShadeSpecModels = ["ggx", "blinn", "hardSpec"] as const;
/** Max. number of per-material overrides. */
export const kStyleShadeMaxOverrides = 64;
const kRecordFloats = 20;

export interface StyleRecord {
    diffuseModel: string;
    wrap: number;
    minnaertK: number;
    specModel: string;
    skyColor: number[] | string;
    groundColor: number[] | string;
    hemiMix: number;
    flatMix: number;
    shininess: number;
    hardThreshold: number;
    hardSoftness: number;
    specScale: number;
}

export const kDefaultStyleRecord: StyleRecord = {
    diffuseModel: "lambert",
    wrap: 0.5,
    minnaertK: 1.2,
    specModel: "ggx",
    skyColor: [0.6, 0.7, 0.9],
    groundColor: [0.25, 0.2, 0.15],
    hemiMix: 0,
    flatMix: 0,
    shininess: 0,
    hardThreshold: 0.5,
    hardSoftness: 0.05,
    specScale: 1,
};
const kRecordKeys = Object.keys(kDefaultStyleRecord) as (keyof StyleRecord)[];

function index(name: string, v: unknown, names: readonly string[]): number {
    const i = typeof v === "number" ? Math.round(v) : names.indexOf(String(v));
    if (i < 0 || i >= names.length) throw new Error(`StyleShade: ${name} must be one of ${names.join("|")} (got ${String(v)})`);
    return i;
}

/** Packs a record into kRecordFloats floats (layout documented in StyleShade.cs.slang). */
export function packStyleRecord(r: StyleRecord, materialId: number): number[] {
    const sky = parseColour(r.skyColor), ground = parseColour(r.groundColor);
    return [
        index("diffuseModel", r.diffuseModel, kStyleShadeDiffuseModels), Number(r.wrap), Number(r.minnaertK), index("specModel", r.specModel, kStyleShadeSpecModels),
        sky[0], sky[1], sky[2], Number(r.hemiMix),
        ground[0], ground[1], ground[2], Number(r.flatMix),
        Number(r.shininess), Number(r.hardThreshold), Number(r.hardSoftness), Number(r.specScale),
        materialId, 0, 0, 0,
    ];
}

/** Rejects ramp / band properties: this pass has no colour-ramp or band-count mode. */
function assertNoRamp(keys: Iterable<string>, where: string): void {
    for (const k of keys) if (/^(ramp|band)/i.test(k)) throw new Error(`StyleShade: '${k}' (${where}) is not supported; StyleShade has no ramp/band mode`);
}

export class StyleShade extends RenderPass {
    private rec: StyleRecord = { ...kDefaultStyleRecord };
    private overrides: ({ materialId: number } & Partial<StyleRecord>)[] = [];
    private styleMix = 1;
    private shadowLight = -1;
    private envLighting = true;
    private analyticLights = true;
    private emissive = true;
    private envBackground = true;

    private pass: ComputePass | null = null;
    private passKey = "";
    private shPass: ComputePass | null = null;
    private shBuffer: Buffer | null = null;
    private shSource: unknown = null;
    private styleBuffer: Buffer | null = null;
    private styleKey = "";

    constructor(device: Device, props: Properties) {
        super(device);
        assertNoRamp([...props.entries()].map(([k]) => k), "property");
        for (const k of kRecordKeys) (this.rec as unknown as Record<string, unknown>)[k] = props.get(k, kDefaultStyleRecord[k] as PropertyValue);
        this.styleMix = Number(props.get("styleMix", 1));
        this.shadowLight = Number(props.get("shadowLight", -1));
        this.envLighting = props.get("envLighting", true);
        this.analyticLights = props.get("analyticLights", true);
        this.emissive = props.get("emissive", true);
        this.envBackground = props.get("envBackground", true);
        const mats = props.get<PropertyValue[]>("materials", []);
        if (!Array.isArray(mats)) throw new Error("StyleShade: materials must be a list of { materialId, ...style }");
        if (mats.length > kStyleShadeMaxOverrides) throw new Error(`StyleShade: at most ${kStyleShadeMaxOverrides} material overrides`);
        for (const m of mats) {
            if (!m || typeof m !== "object" || Array.isArray(m) || typeof (m as Record<string, unknown>)["materialId"] !== "number") {
                throw new Error("StyleShade: every materials entry needs a numeric materialId");
            }
            assertNoRamp(Object.keys(m), "materials entry");
            for (const k of Object.keys(m)) if (k !== "materialId" && !(kRecordKeys as string[]).includes(k)) throw new Error(`StyleShade: unknown style field '${k}'`);
            this.overrides.push({ ...(m as Record<string, unknown>) } as unknown as { materialId: number });
        }
        packStyleRecord(this.rec, -1); // validates the enums and colours now
        for (const o of this.overrides) packStyleRecord({ ...this.rec, ...o }, o.materialId);
    }

    override getProperties(): Properties {
        return new Properties({
            ...(this.rec as unknown as Record<string, PropertyValue>),
            materials: this.overrides.map((o) => ({ ...o })) as unknown as PropertyValue[],
            styleMix: this.styleMix,
            shadowLight: this.shadowLight,
            envLighting: this.envLighting,
            analyticLights: this.analyticLights,
            emissive: this.emissive,
            envBackground: this.envBackground,
        });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Style mix", this.styleMix, 0, 1, 0.01, (v) => (this.styleMix = v));
        ui.dropdown("Diffuse", [...kStyleShadeDiffuseModels], String(this.rec.diffuseModel), (v) => (this.rec.diffuseModel = v));
        ui.slider("Wrap", this.rec.wrap, 0, 1, 0.01, (v) => (this.rec.wrap = v));
        ui.slider("Minnaert k", this.rec.minnaertK, 0.5, 3, 0.01, (v) => (this.rec.minnaertK = v));
        ui.slider("Hemisphere mix", this.rec.hemiMix, 0, 1, 0.01, (v) => (this.rec.hemiMix = v));
        ui.slider("Flat albedo mix", this.rec.flatMix, 0, 1, 0.01, (v) => (this.rec.flatMix = v));
        ui.dropdown("Specular", [...kStyleShadeSpecModels], String(this.rec.specModel), (v) => (this.rec.specModel = v));
        ui.slider("Shininess (0 = from roughness)", this.rec.shininess, 0, 512, 1, (v) => (this.rec.shininess = v));
        ui.slider("Hard spec threshold", this.rec.hardThreshold, 0, 1, 0.01, (v) => (this.rec.hardThreshold = v));
        ui.slider("Hard spec softness", this.rec.hardSoftness, 0, 0.5, 0.005, (v) => (this.rec.hardSoftness = v));
        ui.slider("Specular scale", this.rec.specScale, 0, 4, 0.01, (v) => (this.rec.specScale = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Packed hit information (V-buffer)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("viewW", "View direction (world)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("visibility", "Visibility of the shadowed light (r)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("materialId", "StyleAovs.materialId (raw material id in g)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        const out = (name: string, desc: string, optional: boolean) => {
            const f = r.addOutput(name, desc).texture2D(w, h).format(ResourceFormat.RGBA32Float)
                .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
            if (optional) f.flags(FieldFlags.Optional);
        };
        out("color", "Re-shaded direct radiance", false);
        out("diffuse", "Diffuse radiance (analytic + ambient)", true);
        out("specular", "Specular radiance (analytic + env)", true);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.pass = null;
        this.passKey = "";
        this.shSource = null;
    }

    private updateEnvSH(ctx: RenderContext): void {
        const env = this.scene!.getEnvMap();
        if (this.shBuffer && this.shSource === env) return;
        this.shBuffer ??= new Buffer(this.device, {
            size: 9 * 16, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal, name: "StyleShade::envSH",
        });
        this.shPass ??= ComputePass.create(this.device, { path: kSHShaderFile, defines: this.scene!.getSceneDefines() });
        const root = this.shPass.getRootVar();
        this.scene!.bindShaderData(root);
        root["gSH"] = this.shBuffer;
        this.shPass.execute(ctx, 64, 1);
        this.shSource = env;
    }

    private updateStyles(): number {
        const data = [packStyleRecord(this.rec, -1), ...this.overrides.map((o) => packStyleRecord({ ...this.rec, ...o }, o.materialId))].flat();
        const key = data.join(",");
        const bytes = (kStyleShadeMaxOverrides + 1) * kRecordFloats * 4;
        this.styleBuffer ??= new Buffer(this.device, {
            size: bytes, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "StyleShade::styles",
        });
        if (key !== this.styleKey) {
            const f = new Float32Array(bytes / 4);
            f.set(data);
            this.styleBuffer.setBlob(f);
            this.styleKey = key;
        }
        return data.length / kRecordFloats;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color")!;
        if (!this.scene) {
            ctx.clearTexture(color, [0, 0, 0, 0]);
            return;
        }
        const scene = this.scene;
        const useEnv = this.envLighting && scene.useEnvLight;
        const viewW = rd.getTexture("viewW"), vis = rd.getTexture("visibility"), mat = rd.getTexture("materialId");
        const diff = rd.getTexture("diffuse"), spec = rd.getTexture("specular");
        const defines = scene.getSceneDefines().addAll({
            USE_ANALYTIC_LIGHTS: this.analyticLights && scene.useAnalyticLights ? 1 : 0,
            USE_ENV_LIGHT: useEnv ? 1 : 0,
            USE_ENV_BACKGROUND: this.envBackground && scene.useEnvBackground ? 1 : 0,
            USE_EMISSIVE: this.emissive ? 1 : 0,
            is_valid_gViewW: viewW ? 1 : 0,
            is_valid_gVisibility: vis ? 1 : 0,
            is_valid_gMaterialId: mat ? 1 : 0,
            is_valid_gDiffuse: diff ? 1 : 0,
            is_valid_gSpecular: spec ? 1 : 0,
        });
        const key = defines.key();
        if (!this.pass || key !== this.passKey) {
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.passKey = key;
        }
        if (useEnv) this.updateEnvSH(ctx);
        const records = this.updateStyles();

        let shadowLight = this.shadowLight;
        if (shadowLight < 0) {
            const csm = rd.dictionary.get(kCascadedShadowMapData) as CascadedShadowMapData | undefined;
            shadowLight = csm ? csm.lightIndex : scene.activeLights.findIndex((l) => l.type === LightType.Directional);
        }

        const root = this.pass.getRootVar();
        scene.bindShaderData(root);
        const cb = root["CB"]!;
        cb["gFrameDim"] = [color.width, color.height];
        cb["gShadowLightIndex"] = vis ? shadowLight : -1;
        cb["gRecordCount"] = records;
        cb["gStyleMix"] = Math.min(Math.max(this.styleMix, 0), 1);
        const env = scene.getEnvMap();
        cb["gEnvSpecLodScale"] = env ? Math.max(0, env.texture.mipCount - 1) : 0;
        if (useEnv) root["gEnvSH"] = this.shBuffer!;
        root["gStyles"] = this.styleBuffer!;
        root["gVBuffer"] = rd.getTexture("vbuffer")!;
        root["gViewW"] = viewW ?? dummyTexture(this.device, "ss:viewW");
        root["gVisibility"] = vis ?? dummyTexture(this.device, "ss:vis");
        root["gMaterialId"] = mat ?? dummyTexture(this.device, "ss:mat");
        root["gOutputColor"] = color;
        root["gDiffuse"] = diff ?? dummyTexture(this.device, "ss:outDiff");
        root["gSpecular"] = spec ?? dummyTexture(this.device, "ss:outSpec");
        this.pass.execute(ctx, color.width, color.height);
    }
}

registerRenderPass("StyleShade", (device, props) => new StyleShade(device, props), {
    desc: "Deferred re-shading with a per-material style table (wrap, half-Lambert, Minnaert, hemisphere ambient, flat albedo, Blinn / hard specular)",
});
