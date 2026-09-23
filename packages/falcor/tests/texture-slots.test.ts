/**
 * Per-material texture slot tables (each material class's mTextureSlotInfo)
 * and the OBJ helpers of the assimp importer.
 */

import { describe, expect, it } from "vitest";
import { MaterialType, ShadingModel } from "../src/Scene/Material/MaterialData.js";
import { getTextureSlotSrgb } from "../src/Scene/Material/TextureSlots.js";
import { convertSpecPowerToRoughness, objMaterialLibraries } from "../src/Scene/Importer/FbxImporter.js";

describe("texture slot tables", () => {
    it("decide colour space per material type", () => {
        // StandardMaterial: colour slots are sRGB, packed parameters are not.
        expect(getTextureSlotSrgb(MaterialType.Standard, ShadingModel.MetalRough, "BaseColor")).toBe(true);
        expect(getTextureSlotSrgb(MaterialType.Standard, ShadingModel.MetalRough, "Specular")).toBe(false);
        expect(getTextureSlotSrgb(MaterialType.Standard, ShadingModel.MetalRough, "Transmission")).toBe(true);
        // In spec-gloss mode the specular slot holds a colour.
        expect(getTextureSlotSrgb(MaterialType.Standard, ShadingModel.SpecGloss, "Specular")).toBe(true);
        // Conductors keep eta and k in their colour slots: data, not colour.
        expect(getTextureSlotSrgb(MaterialType.PBRTConductor, ShadingModel.MetalRough, "BaseColor")).toBe(false);
        expect(getTextureSlotSrgb(MaterialType.PBRTCoatedConductor, ShadingModel.MetalRough, "Transmission")).toBe(false);
        expect(getTextureSlotSrgb(MaterialType.PBRTDiffuseTransmission, ShadingModel.MetalRough, "Transmission")).toBe(true);
    });

    it("report slots a material does not have", () => {
        expect(getTextureSlotSrgb(MaterialType.Cloth, ShadingModel.MetalRough, "Emissive")).toBeUndefined();
        expect(getTextureSlotSrgb(MaterialType.Hair, ShadingModel.MetalRough, "Normal")).toBeUndefined();
        expect(getTextureSlotSrgb(MaterialType.PBRTDielectric, ShadingModel.MetalRough, "BaseColor")).toBeUndefined();
        expect(getTextureSlotSrgb(MaterialType.MERL, ShadingModel.MetalRough, "BaseColor")).toBeUndefined();
        // Every basic material has a displacement slot (BasicMaterial's constructor).
        expect(getTextureSlotSrgb(MaterialType.Hair, ShadingModel.MetalRough, "Displacement")).toBe(false);
    });
});

describe("OBJ helpers", () => {
    it("convert a Phong exponent to roughness", () => {
        expect(convertSpecPowerToRoughness(30)).toBeCloseTo(0.25, 12);
        expect(convertSpecPowerToRoughness(0)).toBe(1);
        expect(convertSpecPowerToRoughness(1e9)).toBeCloseTo(0, 3);
    });

    it("find every mtllib reference", () => {
        const obj = "# comment\nmtllib a.mtl\nv 0 0 0\n  mtllib b.mtl c.mtl  \nusemtl x\n";
        expect(objMaterialLibraries(obj)).toEqual(["a.mtl", "b.mtl", "c.mtl"]);
        expect(objMaterialLibraries("v 0 0 0\n")).toEqual([]);
    });
});
