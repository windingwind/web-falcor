/**
 * Per-node animation time quantisation: quantizeTime, ancestor inheritance (resolveNodeQuantization) and the
 * per-node sample time of evaluateGlobals (without `nodeTime` every node samples the global time, as before).
 */

import { describe, it, expect } from "vitest";
import { evaluateGlobals, quantizeTime, resolveNodeQuantization, type AnimationChannel, type SceneNode, type SceneAnimations } from "../src/Scene/Animation/SceneAnimation.js";
import { float3 } from "../src/Utils/Math/Vector.js";
import { quatf } from "../src/Utils/Math/Quaternion.js";

const node = (parent: number, x = 0): SceneNode => ({ parent, t: new float3(x, 0, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) });
const slideX = (nodeID: number): AnimationChannel => ({
    nodeID, path: "translation", interp: "LINEAR", times: new Float32Array([0, 1]), values: new Float32Array([0, 0, 0, 24, 0, 0]),
});

describe("animation time quantisation", () => {
    it("quantizeTime holds frame k at 2*floor(k/2)/fps on twos", () => {
        for (let k = 0; k < 48; k++) expect(quantizeTime(k / 24, { hold: 2 / 24 })).toBeCloseTo(2 * Math.floor(k / 2) / 24, 12);
        for (let k = 0; k < 48; k++) expect(quantizeTime(k / 24, { hold: 3 / 24, phase: 1 / 24 })).toBeCloseTo((3 * Math.floor((k - 1) / 3) + 1) / 24, 12);
        expect(quantizeTime(0.37, { hold: 0 })).toBe(0.37);
    });

    it("descendants inherit the nearest quantised ancestor", () => {
        const nodes = [node(-1), node(0), node(1), node(-1)];
        const r = resolveNodeQuantization(nodes, new Map([[0, { hold: 0.5 }], [2, { hold: 0.25 }]]));
        expect(r.map((q) => q?.hold)).toEqual([0.5, 0.5, 0.25, undefined]);
    });

    it("evaluateGlobals samples each node at its own time; without nodeTime it is unchanged", () => {
        const anim: SceneAnimations = { nodes: [node(-1), node(-1)], channels: [slideX(0), slideX(1)], start: 0, duration: 1 };
        const t = 5 / 24;
        const plain = evaluateGlobals(anim, t);
        const same = evaluateGlobals(anim, t, () => t);
        expect(same.map((m) => m.toArray())).toEqual(plain.map((m) => m.toArray()));
        const q = evaluateGlobals(anim, t, (id) => (id === 0 ? quantizeTime(t, { hold: 2 / 24 }) : t));
        expect(q[0]!.get(0, 3)).toBeCloseTo(4, 4); // held at frame 4 (x = 24 * 4/24)
        expect(q[1]!.get(0, 3)).toBe(plain[1]!.get(0, 3));
    });
});
