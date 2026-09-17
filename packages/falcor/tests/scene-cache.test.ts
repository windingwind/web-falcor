import { describe, expect, it } from "vitest";
import { deserializeScene, serializeScene, type CacheableScene } from "../src/Scene/SceneCache.js";
import { float2, float3, float4 } from "../src/Utils/Math/Vector.js";
import { float4x4 } from "../src/Utils/Math/Matrix.js";
import { quatf } from "../src/Utils/Math/Quaternion.js";
import { AnimationBehavior } from "../src/Scene/Animation/SceneAnimation.js";
import { MaterialType } from "../src/Scene/Material/MaterialData.js";
import { LightType } from "../src/Scene/SceneData.js";

/** One of every v4 scene class with distinctive payloads. */
function makeScene(): CacheableScene {
    const vert = (i: number) => ({
        position: new float3(i, i + 0.5, -i),
        normal: new float3(0, 1, 0),
        tangent: new float4(1, 0, 0, 1),
        texCrd: new float2(i * 0.25, 1 - i * 0.25),
        curveRadius: 0,
    });
    const inverseBind = [float4x4.identity(), new float4x4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]))];
    return {
        meshes: [
            {
                vertices: [vert(0), vert(1), vert(2)],
                indices: new Uint32Array([0, 1, 2]),
                materialID: 0,
                nodeID: 1,
                skin: { boneNodeIDs: [1, 2], inverseBind, boneIDs: new Uint32Array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), weights: new Float32Array([0.5, 0.5, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]) },
                morph: { nodeID: 1, baseWeights: [0.25, 0.75], targets: [{ position: new Float32Array(9).fill(0.125), normal: new Float32Array(9).fill(-1) }, { position: new Float32Array(9).fill(2) }] },
            },
            { vertices: [vert(3), vert(4), vert(5)], indices: new Uint32Array([2, 1, 0]), materialID: 0, transform: new float4x4(new Float32Array([2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1])) },
        ],
        materials: [{ name: "m", header: { materialType: MaterialType.Standard, ior: 1.5 }, basic: { baseColor: new float4(0.3, 0.8, 0.9, 1) } }] as CacheableScene["materials"],
        lights: [{ type: LightType.Point, name: "l", posW: new float3(0, 2, 0), dirW: new float3(0, -1, 0), intensity: new float3(3, 3, 3) }] as CacheableScene["lights"],
        nodes: [
            { parent: -1, t: new float3(0, 0, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
            { parent: 0, t: new float3(1, 0, 0), r: new quatf(0, 0.7071, 0, 0.7071), s: new float3(1, 2, 1) },
            { parent: 0, t: new float3(0, 3, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) },
        ],
        cameraNodeID: 2,
        camera: { position: [0, 1, 5], target: [0, 0, 0], up: [0, 1, 0], focalLength: 21, focalDistance: 10000, apertureRadius: 0 },
        textures: [{ png: new Uint8Array([1, 2, 3, 4, 5]), srgb: true }],
        curves: [{ positionsRadii: new Float32Array([0, 0, 0, 0.1, 1, 0, 0, 0.2]), texCrds: null, indices: new Uint32Array([0]), materialID: 0 }],
        envMap: { bytes: new Uint8Array([9, 8, 7]), isExr: false, intensity: 1.5, tint: [1, 0.5, 0.25], rotationDeg: [0, 90, 0] },
        animations: [
            { nodeID: 1, path: "rotation", times: new Float32Array([0, 1, 2]), values: new Float32Array(12).map((_, i) => i / 12), interp: "LINEAR", clip: 0, preInfinity: AnimationBehavior.Cycle, postInfinity: AnimationBehavior.Oscillate },
            { nodeID: 2, path: "translation", times: new Float32Array([0.5]), values: new Float32Array([1, 2, 3]), interp: "STEP" },
        ],
        weightTracks: [{ nodeID: 1, times: new Float32Array([0, 2]), values: new Float32Array([1, 0, 0, 1]), numTargets: 2, interp: "LINEAR" }],
        sdfGrids: {
            recipes: [
                { type: "ndsdf", narrowBandThickness: 2.5, brickWidth: 7, ops: [{ kind: "cheese", gridWidth: 32, seed: 7 }] },
                // A grid loaded from a `.sdfg` file carries its corner values.
                { type: "svs", narrowBandThickness: 1, brickWidth: 4, ops: [{ kind: "values", gridWidth: 2, values: new Float32Array(27).map((_v, i) => i / 27) }] },
            ],
            instances: [{ gridIndex: 0, materialID: 0, transform: float4x4.identity() }, { gridIndex: 0, materialID: 0 }],
        },
        gridVolumes: [{ name: "smoke", densityScale: 0.5, emissionScale: 1, albedo: [0.5, 0.5, 0.5], anisotropy: 0.1, emissionTemperature: 0, grids: [{ slot: "density", bytes: new Uint8Array([7, 7, 7, 7, 7, 7, 7]) }] }],
    };
}

describe("SceneCache v4 serialization", () => {
    it("round-trips every scene class bit-exactly", () => {
        const scene = makeScene();
        const bytes = serializeScene(scene);
        expect(bytes.byteLength % 1).toBe(0);
        const back = deserializeScene(bytes);

        // Plain JSON view of everything but typed arrays / math types.
        const json = (v: unknown) => JSON.stringify(v, (_k, x) => (x instanceof Float32Array || x instanceof Uint32Array || x instanceof Uint8Array ? Array.from(x) : x));
        expect(json(back.camera)).toBe(json(scene.camera));
        expect(back.cameraNodeID).toBe(2);
        expect(json(back.nodes)).toBe(json(scene.nodes));
        expect(json(back.materials)).toBe(json(scene.materials));
        expect(json(back.lights)).toBe(json(scene.lights));
        expect(json(back.textures)).toBe(json(scene.textures));
        expect(json(back.curves)).toBe(json(scene.curves));
        expect(json(back.envMap)).toBe(json(scene.envMap));
        expect(json(back.animations)).toBe(json(scene.animations));
        expect(json(back.weightTracks)).toBe(json(scene.weightTracks));
        expect(json(back.sdfGrids)).toBe(json(scene.sdfGrids));
        // The corner values must come back as a typed array, not a plain object.
        const valuesOp = back.sdfGrids.recipes[1]!.ops[0]!;
        expect(valuesOp.kind).toBe("values");
        expect((valuesOp as { values: Float32Array }).values).toBeInstanceOf(Float32Array);
        expect(json(back.gridVolumes)).toBe(json(scene.gridVolumes));
        expect(back.meshes.length).toBe(2);
        expect(json(back.meshes[0]!.vertices)).toBe(json(scene.meshes[0]!.vertices));
        expect(Array.from(back.meshes[1]!.indices)).toEqual([2, 1, 0]);
        expect(json(back.meshes[0]!.skin)).toBe(json(scene.meshes[0]!.skin));
        expect(json(back.meshes[0]!.morph)).toBe(json(scene.meshes[0]!.morph));
        expect(back.meshes[0]!.morph!.targets[1]!.normal).toBeUndefined();
        expect(back.meshes[1]!.skin).toBeUndefined();
        expect(json(back.meshes[1]!.transform)).toBe(json(scene.meshes[1]!.transform));
        expect(back.meshes[0]!.transform).toBeUndefined();
        // Typed arrays are copies, not views into the file buffer.
        expect(back.animations[0]!.times.buffer).not.toBe(bytes.buffer);
    });

    it("rejects other versions", () => {
        const bytes = serializeScene(makeScene());
        new DataView(bytes.buffer).setUint32(4, 3, true);
        expect(() => deserializeScene(bytes)).toThrow(/magic\/version/);
    });
});
