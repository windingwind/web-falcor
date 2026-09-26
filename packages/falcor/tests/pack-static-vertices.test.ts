/** packStaticVertices reads packed stores directly; its bytes must equal the getter-based packing. */

import { describe, expect, it } from "vitest";
import { createPackedVertices, encodeNormal2x16, kPackedVertexFloats, packStaticVertices, type StaticVertex } from "../src/Scene/SceneData.js";
import { f32tof16 } from "../src/Scene/Material/MaterialData.js";
import { float2, float3, float4 } from "../src/Utils/Math/Vector.js";

/** The previous implementation (one getter call per component). */
function reference(vertices: StaticVertex[]): ArrayBuffer {
    const buffer = new ArrayBuffer(vertices.length * 48);
    const dv = new DataView(buffer);
    vertices.forEach((v, i) => {
        const base = i * 48;
        dv.setFloat32(base, v.position.x, true); dv.setFloat32(base + 4, v.position.y, true); dv.setFloat32(base + 8, v.position.z, true);
        const [nx, ny, nz] = [f32tof16(v.normal.x), f32tof16(v.normal.y), f32tof16(v.normal.z)];
        let sign = Math.fround(v.tangent.w);
        if ((v.curveRadius ?? 0) > 0) sign *= Math.fround(v.curveRadius!);
        dv.setUint32(base + 16, ((ny << 16) | nx) >>> 0, true);
        dv.setUint32(base + 20, ((f32tof16(sign) << 16) | nz) >>> 0, true);
        dv.setUint32(base + 24, encodeNormal2x16(new float3(v.tangent.x, v.tangent.y, v.tangent.z)) >>> 0, true);
        dv.setFloat32(base + 32, v.texCrd.x, true); dv.setFloat32(base + 36, v.texCrd.y, true);
    });
    return buffer;
}

describe("packStaticVertices", () => {
    it("matches the getter-based packing for packed and plain vertices", () => {
        let seed = 7;
        const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647) * 4 - 2;
        const packed = createPackedVertices(500, Float32Array.from({ length: 500 * kPackedVertexFloats }, rnd));
        const plain: StaticVertex[] = Array.from({ length: 500 }, (_v, i) => ({
            position: new float3(rnd(), rnd(), rnd()), normal: new float3(rnd(), rnd(), rnd()), tangent: new float4(rnd(), rnd(), rnd(), i % 2 ? 1 : -1),
            texCrd: new float2(rnd(), rnd()), curveRadius: i % 3 ? undefined : Math.abs(rnd()),
        }));
        for (const vs of [packed, plain]) expect(new Uint8Array(packStaticVertices(vs))).toEqual(new Uint8Array(reference(vs)));
    });
});
