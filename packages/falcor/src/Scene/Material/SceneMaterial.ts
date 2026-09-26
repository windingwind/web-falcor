/**
 * A scene's material: the SceneMaterialDesc record plus native's Material/BasicMaterial python
 * properties (baseColor, roughness, emissiveColor, alphaMode, ...). Each edit updates the record
 * and repacks the material, as MaterialSystem::update does for BasicMaterial::markUpdates.
 */

import { float2, float3, float4 } from "../../Utils/Math/Vector.js";
import type { SceneMaterialDesc } from "../Scene.js";
import { AlphaMode, MaterialType, ShadingModel, type BasicMaterialDesc, type MaterialHeaderDesc } from "./MaterialData.js";
import type { MERLBRDF, MERLMixData } from "./MERLFile.js";
import type { RGLMeasurement } from "./RGLFile.js";

type Vec = { x: number; y: number; z: number; w?: number };
const f3 = (v: Vec) => new float3(Number(v.x), Number(v.y), Number(v.z));
const f4 = (v: Vec) => new float4(Number(v.x), Number(v.y), Number(v.z), Number(v.w ?? 1));

export class SceneMaterial implements SceneMaterialDesc {
    name?: string;
    header?: Partial<MaterialHeaderDesc>;
    basic: BasicMaterialDesc;
    merl?: MERLBRDF;
    rgl?: RGLMeasurement;
    merlMix?: MERLMixData;

    constructor(
        desc: SceneMaterialDesc,
        private readonly changed: (m: SceneMaterial) => void = () => {},
        private readonly derivedAlphaMode: (header: Partial<MaterialHeaderDesc>, basic: BasicMaterialDesc) => AlphaMode = () => AlphaMode.Opaque,
    ) {
        this.name = desc.name;
        this.header = desc.header;
        this.basic = desc.basic;
        if (desc.merl) this.merl = desc.merl;
        if (desc.rgl) this.rgl = desc.rgl;
        if (desc.merlMix) this.merlMix = desc.merlMix;
    }

    private setHeader(patch: Partial<MaterialHeaderDesc>): void {
        this.header = { ...this.header, ...patch };
        this.changed(this);
    }
    private setBasic(patch: Partial<BasicMaterialDesc>): void {
        Object.assign(this.basic, patch);
        // BasicMaterial::updateEmissiveFlag.
        const e = this.basic.emissive, k = this.basic.emissiveFactor ?? 1;
        const emissive = this.basic.texEmissive !== undefined || (!!e && k > 0 && (e.x > 0 || e.y > 0 || e.z > 0));
        this.header = { ...this.header, emissive };
        this.changed(this);
    }
    private get specular(): float4 {
        return this.basic.specular ?? new float4(0, 0.5, 0, 0);
    }

    /** Mirrors Material::getType. */
    get type(): MaterialType { return this.header?.materialType ?? MaterialType.Standard; }
    get shadingModel(): ShadingModel { return this.basic.shadingModel ?? ShadingModel.MetalRough; }
    get baseColor(): float4 { return this.basic.baseColor ?? new float4(1, 1, 1, 1); }
    set baseColor(v: float4) { this.setBasic({ baseColor: f4(v) }); }
    get specularParams(): float4 { return this.specular; }
    set specularParams(v: float4) { this.setBasic({ specular: f4(v) }); }
    /**
     * Per material type as natively: PBRT conductor/dielectric/coated-diffuse roughness is specular.rg (float2), the
     * coated conductor's all of specular (float4), Cloth's specular.g; Standard's specular.g, or 0 unless metal-rough.
     */
    get roughness(): number | float2 | float4 {
        const s = this.specular;
        switch (this.type) {
            case MaterialType.PBRTConductor:
            case MaterialType.PBRTDielectric:
            case MaterialType.PBRTCoatedDiffuse:
                return new float2(s.x, s.y);
            case MaterialType.PBRTCoatedConductor:
                return new float4(s.x, s.y, s.z, s.w);
            case MaterialType.Cloth:
                return s.y;
            default:
                return this.shadingModel === ShadingModel.MetalRough ? s.y : 0;
        }
    }
    set roughness(v: number | Vec) {
        const s = this.specular;
        if (typeof v === "number") this.setBasic({ specular: new float4(s.x, Number(v), s.z, s.w) });
        else if (v.w !== undefined && this.type === MaterialType.PBRTCoatedConductor) this.setBasic({ specular: f4(v) });
        else this.setBasic({ specular: new float4(Number(v.x), Number(v.y), s.z, s.w) });
    }
    /** StandardMaterial::getMetallic: specular.b, or 0 unless metal-rough. */
    get metallic(): number { return this.shadingModel === ShadingModel.MetalRough ? this.specular.z : 0; }
    set metallic(v: number) { const s = this.specular; this.setBasic({ specular: new float4(s.x, s.y, Number(v), s.w) }); }
    get transmissionColor(): float3 { return this.basic.transmission ?? new float3(1, 1, 1); }
    set transmissionColor(v: float3) { this.setBasic({ transmission: f3(v) }); }
    get diffuseTransmission(): number { return this.basic.diffuseTransmission ?? 0; }
    set diffuseTransmission(v: number) { this.setBasic({ diffuseTransmission: Number(v) }); }
    get specularTransmission(): number { return this.basic.specularTransmission ?? 0; }
    set specularTransmission(v: number) { this.setBasic({ specularTransmission: Number(v) }); }
    get indexOfRefraction(): number { return this.header?.ior ?? 1.5; }
    set indexOfRefraction(v: number) { this.setHeader({ ior: Number(v) }); }
    get emissiveColor(): float3 { return this.basic.emissive ?? new float3(0, 0, 0); }
    set emissiveColor(v: float3) { this.setBasic({ emissive: f3(v) }); }
    get emissiveFactor(): number { return this.basic.emissiveFactor ?? 1; }
    set emissiveFactor(v: number) { this.setBasic({ emissiveFactor: Number(v) }); }
    /** Set explicitly, else derived as BasicMaterial::updateAlphaMode does. */
    get alphaMode(): AlphaMode { return this.header?.alphaMode ?? this.derivedAlphaMode(this.header ?? {}, this.basic); }
    set alphaMode(v: AlphaMode) { this.setHeader({ alphaMode: Number(v) as AlphaMode }); }
    get alphaThreshold(): number { return this.header?.alphaThreshold ?? 0.5; }
    set alphaThreshold(v: number) { this.setHeader({ alphaThreshold: Number(v) }); }
    /** Material::isDoubleSided after BasicMaterial::adjustDoubleSidedFlag (transmission or displacement force it on). */
    get doubleSided(): boolean {
        const b = this.basic;
        return !!this.header?.doubleSided || (b.diffuseTransmission ?? 0) > 0 || (b.specularTransmission ?? 0) > 0 || b.texDisplacement !== undefined;
    }
    set doubleSided(v: boolean) { this.setHeader({ doubleSided: Boolean(v) }); }
    get thinSurface(): boolean { return this.header?.thinSurface ?? false; }
    set thinSurface(v: boolean) { this.setHeader({ thinSurface: Boolean(v) }); }
    get nestedPriority(): number { return this.header?.nestedPriority ?? 0; }
    set nestedPriority(v: number) { this.setHeader({ nestedPriority: Number(v) }); }
    get volumeAbsorption(): float3 { return this.basic.volumeAbsorption ?? new float3(0, 0, 0); }
    set volumeAbsorption(v: float3) { this.setBasic({ volumeAbsorption: f3(v) }); }
    get volumeScattering(): float3 { return this.basic.volumeScattering ?? new float3(0, 0, 0); }
    set volumeScattering(v: float3) { this.setBasic({ volumeScattering: f3(v) }); }
    get volumeAnisotropy(): number { return this.basic.volumeAnisotropy ?? 0; }
    set volumeAnisotropy(v: number) { this.setBasic({ volumeAnisotropy: Number(v) }); }
    get displacementScale(): number { return this.basic.displacementScale ?? 0; }
    set displacementScale(v: number) { this.setBasic({ displacementScale: Number(v) }); }
    get displacementOffset(): number { return this.basic.displacementOffset ?? 0; }
    set displacementOffset(v: number) { this.setBasic({ displacementOffset: Number(v) }); }

    /** Material::setRoughnessMollification: a no-op in native's materials too. */
    setRoughnessMollification(_value: number): void {}

    /** Mirrors Material::clearTexture(slot) (BaseColor, Specular, Emissive, Normal, Transmission, Displacement). */
    clearTexture(slot: string | { name?: string; value?: string }): void {
        const key = typeof slot === "string" ? slot : (slot.name ?? slot.value ?? String(slot));
        const field = ({ BaseColor: "texBaseColor", Specular: "texSpecular", Emissive: "texEmissive", Normal: "texNormalMap", Transmission: "texTransmission", Displacement: "texDisplacement" } as const)[key.replace(/^.*\./, "") as "BaseColor"];
        if (field) this.setBasic({ [field]: undefined });
    }
}
