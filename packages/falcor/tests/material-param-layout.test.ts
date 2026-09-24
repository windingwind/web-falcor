/** MaterialParamLayout: native's parameter offsets, float16 serialization and clamped deserialization. */
import { describe, it, expect } from "vitest";
import { deserializeMaterialParams, getMaterialParamLayout, kMaterialParamCount, serializeMaterialParams } from "../src/Scene/Material/MaterialParamLayout.js";
import { MaterialType } from "../src/Scene/Material/MaterialData.js";
import { float3, float4 } from "../src/Utils/Math/Vector.js";

describe("MaterialParamLayout", () => {
    it("serializes a MetalRough standard material at native's offsets", () => {
        const m = { header: { materialType: MaterialType.Standard, ior: 1.5 }, basic: { baseColor: new float4(0.5, 0.9, 0.2, 1), specular: new float4(0, 0.2, 0.6, 0), emissive: new float3(1, 2, 3), emissiveFactor: 2 } };
        const p = serializeMaterialParams(m);
        expect(p.length).toBe(kMaterialParamCount);
        expect(Array.from(p.slice(0, 6))).toEqual([0.5, 0.89990234375, 0.199951171875, 0.60009765625, 0.199951171875, 1.5]);
        expect(Array.from(p.slice(11, 15))).toEqual([1, 2, 3, 2]);
        expect(getMaterialParamLayout(m).map((e) => `${e.pythonName}@${e.offset}x${e.size}`)).toContain("roughness@4x1");
    });

    it("clamps like detail::clampMaterialParam and keeps alpha", () => {
        const m = { header: { materialType: MaterialType.PBRTConductor }, basic: { baseColor: new float4(0.5, 0.5, 0.5, 0.25) } };
        const p = new Float32Array(kMaterialParamCount);
        p.set([2, -1, 0.5, 0.2, 0.3, 0.4, 0.01, 0.9]);
        deserializeMaterialParams(m, p);
        const b = m.basic as { baseColor: float4; specular: float4 };
        expect([b.baseColor.x, b.baseColor.y, b.baseColor.w]).toEqual([1 - 1e-4, 1e-4, 0.25]);
        // Conductor roughness has a 0.05 floor.
        expect([b.specular.x, b.specular.y]).toEqual([0.05, Math.fround(0.9)]);
    });
});
