/**
 * BasicMaterialData.volumeAnisotropy: packed as float16 after volumeAbsorption (payload offset 60),
 * clamped to +-0.99 like BasicMaterial::setVolumeAnisotropy.
 */
import { describe, expect, it } from "vitest";
import { MaterialType, packBasicMaterialBlob } from "../src/Scene/Material/MaterialData.js";
import { float16ToFloat32 } from "../src/Utils/Math/Float16.js";

const anisotropyAt = (g: number) => {
    const blob = packBasicMaterialBlob({ materialType: MaterialType.Standard }, { volumeAnisotropy: g });
    return float16ToFloat32(new DataView(blob.buffer).getUint16(16 + 60, true));
};

describe("BasicMaterialData", () => {
    it("packs volumeAnisotropy", () => {
        expect(anisotropyAt(0)).toBe(0);
        expect(anisotropyAt(0.5)).toBe(0.5);
        expect(anisotropyAt(-0.25)).toBe(-0.25);
    });
    it("clamps volumeAnisotropy to +-0.99", () => {
        expect(anisotropyAt(5)).toBeCloseTo(0.99, 3);
        expect(anisotropyAt(-5)).toBeCloseTo(-0.99, 3);
    });
});
