/**
 * Per-material texture slot tables, mirroring each material class's
 * `mTextureSlotInfo` (StandardMaterial.cpp, ClothMaterial.cpp, HairMaterial.cpp,
 * the PBRT materials, MERLMixMaterial.cpp, plus BasicMaterial's Displacement).
 *
 * A slot a material does not declare is ignored with a warning, exactly as
 * MaterialTextureLoader::loadTexture does, and the table's sRGB flag decides
 * the colour space the texture is decoded in.
 */

import { MaterialType, ShadingModel } from "./MaterialData.js";

export type TextureSlotName = "BaseColor" | "Specular" | "Emissive" | "Normal" | "Transmission" | "Displacement" | "Index";

/** Slot -> sRGB, for the slots a material declares. */
type SlotTable = Partial<Record<TextureSlotName, boolean>>;

/** BasicMaterial's constructor adds Displacement to every basic material. */
const kBasic: SlotTable = { Displacement: false };

function tableFor(type: MaterialType, shadingModel: ShadingModel): SlotTable {
    switch (type) {
        case MaterialType.Standard:
            return {
                ...kBasic,
                BaseColor: true,
                // Metal-rough packs roughness/metallic (linear); spec-gloss stores a specular colour.
                Specular: shadingModel === ShadingModel.SpecGloss,
                Normal: false,
                Emissive: true,
                Transmission: true,
            };
        case MaterialType.Cloth:
            return { ...kBasic, BaseColor: true, Specular: false, Normal: false };
        case MaterialType.Hair:
            return { ...kBasic, BaseColor: true, Specular: false };
        case MaterialType.PBRTDiffuse:
            return { ...kBasic, BaseColor: true, Normal: false };
        case MaterialType.PBRTCoatedDiffuse:
            return { ...kBasic, BaseColor: true, Specular: false, Normal: false };
        case MaterialType.PBRTDiffuseTransmission:
            return { ...kBasic, BaseColor: true, Transmission: true, Normal: false };
        case MaterialType.PBRTDielectric:
            return { ...kBasic, Specular: false, Normal: false };
        case MaterialType.PBRTConductor:
        case MaterialType.PBRTCoatedConductor:
            // The base colour holds eta and the transmission colour k: data, not colour.
            return { ...kBasic, BaseColor: false, Transmission: false, Specular: false, Normal: false };
        case MaterialType.MERLMix:
            return { Normal: false, Index: false };
        default:
            return {}; // MERL, RGL: measured data only
    }
}

/**
 * Mirrors Material::getTextureSlotInfo for one slot.
 *
 * @returns undefined when the material has no such slot, else whether it is sRGB.
 */
export function getTextureSlotSrgb(type: MaterialType, shadingModel: ShadingModel, slot: string): boolean | undefined {
    return tableFor(type, shadingModel)[slot as TextureSlotName];
}
