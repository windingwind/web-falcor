import { describe, expect, it } from "vitest";

describe("pretransformStaticMeshes", () => {
    it("transforms packed vertices in place exactly as plain vertices", async () => {
        const { pretransformStaticMeshes } = await import("../src/Scene/SceneBuilder.js");
        const { createPackedVertices, copyVertex } = await import("../src/Scene/SceneData.js");
        const { float4x4 } = await import("../src/Utils/Math/Matrix.js");
        let seed = 7;
        const rnd = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648) * 2 - 1;
        const n = 500;
        const data = new Float32Array(n * 12).map(() => rnd());
        const packed = createPackedVertices(n, data);
        const plain = packed.map((v) => copyVertex(v));
        // A mirroring, non-uniformly scaled, translated transform.
        const m = new float4x4(new Float32Array([-1.5, 0.2, 0.1, 3, 0.3, 0.7, -0.4, -2, 0.05, 0.6, 2.1, 0.5, 0, 0, 0, 1]));
        const mesh = (vertices: typeof packed) => ({ vertices, indices: Uint32Array.from({ length: n - (n % 3) }, (_v, i) => i), materialID: 0, transform: m });
        const [a, b] = [mesh(packed), mesh(plain)];
        pretransformStaticMeshes([a, b] as never, [], []);
        const words = (v: typeof packed) => v.flatMap((x) => [...x.position.toArray(), ...x.normal.toArray(), x.tangent.x, x.tangent.y, x.tangent.z, x.tangent.w, x.texCrd.x, x.texCrd.y]);
        expect(words(a.vertices)).toEqual(words(b.vertices));
        expect(Array.from(a.indices)).toEqual(Array.from(b.indices));
        expect(a.transform).toBeUndefined();
    }, 60000);
});
