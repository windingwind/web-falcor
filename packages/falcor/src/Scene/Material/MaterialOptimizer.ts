/**
 * MaterialSystem::optimizeMaterials / BasicMaterial::optimizeTexture: textures whose used
 * channels are constant become the material's uniform values and leave the material
 * (Bistro's 16x16 constant spec maps become specular params, for instance). The
 * TextureAnalyzer result (min/max per channel, the constant value) comes from a CPU scan
 * of each texture's decoded image, sRGB-decoded like a GPU load.
 */
import { float3, float4 } from "../../Utils/Math/Vector.js";
import type { SceneMaterialDesc } from "../Scene.js";
import { MaterialType, ShadingModel, TextureHandleMode, packBasicMaterialBlob } from "./MaterialData.js";
import type { TextureManager } from "./TextureManager.js";

/** Channel masks as TextureChannelFlags bits: R 1, G 2, B 4, A 8. */
const RGB = 7;
const RGBA = 15;

/** Channels a slot uses (mTextureSlotInfo masks); non-Standard materials need all four constant. */
function slotMask(type: MaterialType, shadingModel: ShadingModel, slot: "BaseColor" | "Specular" | "Emissive" | "Transmission"): number {
    if (type !== MaterialType.Standard) return RGBA;
    if (slot === "Specular") return shadingModel === ShadingModel.SpecGloss ? RGBA : 2 | 4; // metal-rough: roughness G, metallic B
    return slot === "BaseColor" ? RGBA : RGB;
}

const textureID = (handle: number | undefined) => (handle !== undefined && ((handle >>> 29) & 0x3) === TextureHandleMode.Texture ? handle & 0x1fffffff : undefined);

/** Replaces constant material textures by uniform values, in place; returns how many were removed. */
export function optimizeMaterialTextures(materials: SceneMaterialDesc[], textureManager: TextureManager): number {
    let removed = 0;
    for (const m of materials) {
        const type = m.header?.materialType ?? MaterialType.Standard;
        if (type === MaterialType.MERL || type === MaterialType.MERLMix || type === MaterialType.RGL) continue;
        const shadingModel = m.basic.shadingModel ?? ShadingModel.MetalRough;
        const b = m.basic;
        const analyze = (handle: number | undefined) => {
            const id = textureID(handle);
            return id === undefined ? undefined : textureManager.analyze(id);
        };

        const base = analyze(b.texBaseColor);
        if (base) {
            // Hair has no alpha channel in its base color slot; alpha needs a format with alpha.
            const hasAlpha = type !== MaterialType.Hair && base.hasAlpha;
            const colorConstant = base.isConstant(RGB);
            const alphaConstant = base.isConstant(8);
            let color = b.baseColor ?? new float4(1, 1, 1, 1);
            if (colorConstant) color = new float4(base.value[0]!, base.value[1]!, base.value[2]!, color.w);
            if (hasAlpha && alphaConstant) color = new float4(color.x, color.y, color.z, base.value[3]!);
            b.baseColor = color;
            if (colorConstant && (!hasAlpha || alphaConstant)) {
                b.texBaseColor = undefined;
                removed++;
            }
        }
        const spec = analyze(b.texSpecular);
        if (spec?.isConstant(slotMask(type, shadingModel, "Specular"))) {
            b.texSpecular = undefined;
            b.specular = new float4(spec.value[0]!, spec.value[1]!, spec.value[2]!, spec.value[3]!);
            removed++;
        }
        const emissive = analyze(b.texEmissive);
        if (emissive?.isConstant(slotMask(type, shadingModel, "Emissive"))) {
            b.texEmissive = undefined;
            b.emissive = new float3(emissive.value[0]!, emissive.value[1]!, emissive.value[2]!);
            removed++;
        }
        const transmission = analyze(b.texTransmission);
        if (transmission?.isConstant(slotMask(type, shadingModel, "Transmission"))) {
            b.texTransmission = undefined;
            b.transmission = new float3(transmission.value[0]!, transmission.value[1]!, transmission.value[2]!);
            removed++;
        }
    }
    return removed;
}

/**
 * MaterialSystem::removeDuplicateMaterials: materials equal in everything but their name merge
 * into the first one, in place; returns the old-to-new ID map. Measured materials never merge.
 */
export function removeDuplicateMaterials(materials: SceneMaterialDesc[]): number[] {
    const unique: SceneMaterialDesc[] = [];
    const keys: string[] = [];
    const idMap = materials.map((m) => {
        const type = m.header?.materialType ?? MaterialType.Standard;
        const measured = m.merl || m.rgl || m.merlMix || type === MaterialType.MERL || type === MaterialType.MERLMix || type === MaterialType.RGL;
        const key = measured ? "" : Array.from(packBasicMaterialBlob({ materialType: MaterialType.Standard, ...m.header }, m.basic)).join(",");
        const found = measured ? -1 : keys.indexOf(key);
        if (found >= 0) return found;
        unique.push(m);
        keys.push(measured ? `\0${unique.length}` : key);
        return unique.length - 1;
    });
    materials.splice(0, materials.length, ...unique);
    return idMap;
}
