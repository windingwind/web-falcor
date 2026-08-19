/**
 * Scene animation interpolation: LINEAR, STEP, and glTF CUBICSPLINE (Hermite)
 * keyframe sampling through the public evaluateGlobals() API.
 */

import { describe, it, expect } from "vitest";
import { evaluateGlobals, applyMorph, sampleMorphWeights, type AnimationChannel, type SceneNode, type SceneAnimations, type MorphDesc, type WeightTrack } from "../src/Scene/Animation/SceneAnimation.js";
import { float4 } from "../src/Utils/Math/Vector.js";
import { float3 } from "../src/Utils/Math/Vector.js";
import { quatf } from "../src/Utils/Math/Quaternion.js";

const closeTo = (a: number, b: number, eps = 1e-4) => expect(Math.abs(a - b)).toBeLessThan(eps);

/** One root node driven by a single translation channel; returns its world x/y/z. */
function evalTranslation(ch: AnimationChannel, time: number): [number, number, number] {
    const nodes: SceneNode[] = [{ parent: -1, t: new float3(0, 0, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) }];
    const anim: SceneAnimations = { nodes, channels: [ch], start: 0, duration: 1 };
    const g = evaluateGlobals(anim, time)[0]!;
    return [g.get(0, 3), g.get(1, 3), g.get(2, 3)];
}

describe("SceneAnimation interpolation", () => {
    it("LINEAR lerps between keyframes", () => {
        const ch: AnimationChannel = {
            nodeID: 0, path: "translation", interp: "LINEAR",
            times: new Float32Array([0, 1]),
            values: new Float32Array([0, 0, 0, 10, 0, 0]),
        };
        closeTo(evalTranslation(ch, 0.5)[0], 5);
        closeTo(evalTranslation(ch, 0.25)[0], 2.5);
    });

    it("STEP holds the previous keyframe", () => {
        const ch: AnimationChannel = {
            nodeID: 0, path: "translation", interp: "STEP",
            times: new Float32Array([0, 1]),
            values: new Float32Array([0, 0, 0, 10, 0, 0]),
        };
        closeTo(evalTranslation(ch, 0.9)[0], 0);
        closeTo(evalTranslation(ch, 1.0)[0], 10);
    });

    // CUBICSPLINE output stores [inTangent, value, outTangent] per keyframe.
    it("CUBICSPLINE is exact at keyframes and Hermite-smooth between", () => {
        // Zero tangents -> smooth ease (h00*v0 + h01*v1), distinct from linear.
        const ease: AnimationChannel = {
            nodeID: 0, path: "translation", interp: "CUBICSPLINE",
            times: new Float32Array([0, 1]),
            values: new Float32Array([
                0, 0, 0, /*in*/ 0, 0, 0 /*value*/, 0, 0, 0, /*out*/
                0, 0, 0, /*in*/ 1, 0, 0 /*value*/, 0, 0, 0, /*out*/
            ]),
        };
        closeTo(evalTranslation(ease, 0)[0], 0); // exact at start
        closeTo(evalTranslation(ease, 1)[0], 1); // exact at end
        // Hermite basis at f=0.25 with zero tangents: h01 = -2t^3+3t^2 = 0.15625.
        closeTo(evalTranslation(ease, 0.25)[0], 0.15625);
        // Midpoint stays 0.5 (h00*0 + h01*1 with h00=h01=0.5).
        closeTo(evalTranslation(ease, 0.5)[0], 0.5);
    });

    it("CUBICSPLINE tangents bend the curve (analytic check)", () => {
        // v0=0 (out=6), v1=0 (in=6): p(t)=dt*(h10*b0 + h11*a1), symmetric hump.
        const hump: AnimationChannel = {
            nodeID: 0, path: "translation", interp: "CUBICSPLINE",
            times: new Float32Array([0, 1]),
            values: new Float32Array([
                0, 0, 0, /*in*/ 0, 0, 0 /*value*/, 6, 0, 0, /*out*/
                6, 0, 0, /*in*/ 0, 0, 0 /*value*/, 0, 0, 0, /*out*/
            ]),
        };
        // f=0.5: h10 = t^3-2t^2+t = 0.125, h11 = t^3-t^2 = -0.125; x = 0.125*6 + (-0.125)*6 = 0.
        closeTo(evalTranslation(hump, 0.5)[0], 0);
        // f=0.25: h10 = 0.140625, h11 = -0.046875; x = 6*(0.140625) + 6*(-0.046875) = 0.5625.
        closeTo(evalTranslation(hump, 0.25)[0], 0.5625);
    });
});

describe("SceneAnimation morph targets", () => {
    const vert = (x: number, y: number, z: number) => ({ position: new float3(x, y, z), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: { x: 0, y: 0 } as never });
    const morph: MorphDesc = {
        nodeID: 0,
        baseWeights: [0],
        // one target pushes only the top vertex up by +2 in Y.
        targets: [{ position: new Float32Array([0, 0, 0, 0, 2, 0]) }],
    };

    it("blends target deltas by weight (top moves, base fixed)", () => {
        const bind = [vert(0, 0, 0), vert(0, 1, 0)];
        const w0 = applyMorph(bind, morph, [0]);
        closeTo(w0[1]!.position.y, 1); // weight 0 -> unchanged
        const w1 = applyMorph(bind, morph, [1]);
        closeTo(w1[1]!.position.y, 3); // weight 1 -> 1 + 2
        closeTo(w1[0]!.position.y, 0); // base vertex has zero delta
        const wh = applyMorph(bind, morph, [0.5]);
        closeTo(wh[1]!.position.y, 2); // half weight
    });

    it("samples an animated weight track (LINEAR)", () => {
        const track: WeightTrack = { nodeID: 0, numTargets: 1, interp: "LINEAR", times: new Float32Array([0, 1]), values: new Float32Array([0, 1]) };
        closeTo(sampleMorphWeights(morph, [track], 0)[0]!, 0);
        closeTo(sampleMorphWeights(morph, [track], 0.5)[0]!, 0.5);
        closeTo(sampleMorphWeights(morph, [track], 1)[0]!, 1);
        // no track for this node -> falls back to base weights.
        expect(sampleMorphWeights(morph, [], 0.5)).toEqual([0]);
    });
});

describe("SceneAnimation pre/post-infinity behaviors", () => {
    // Keys at t=1..3: x 0 -> 10 (slope 5/s), mirrors Animation::calcSampleTime cases.
    const mk = (pre?: number, post?: number): AnimationChannel => ({
        nodeID: 0, path: "translation", interp: "LINEAR",
        times: new Float32Array([1, 3]),
        values: new Float32Array([0, 0, 0, 10, 0, 0]),
        preInfinity: pre, postInfinity: post,
    });

    it("Constant clamps outside the key range (default)", () => {
        closeTo(evalTranslation(mk(), 0)[0], 0);
        closeTo(evalTranslation(mk(), 4)[0], 10);
    });

    it("Cycle wraps relative to the first keyframe (both sides)", () => {
        const ch = mk(2, 2); // Behavior.Cycle
        closeTo(evalTranslation(ch, 0)[0], 5); // t=0 -> wrapped 2
        closeTo(evalTranslation(ch, 4)[0], 5); // t=4 -> wrapped 2
        closeTo(evalTranslation(ch, 4.5)[0], 7.5); // t=4.5 -> wrapped 2.5
    });

    it("Oscillate ping-pongs over 2x duration", () => {
        const ch = mk(3, 3); // Behavior.Oscillate
        closeTo(evalTranslation(ch, 4)[0], 5); // offset 3 -> mirrored 1 -> t=2
        closeTo(evalTranslation(ch, 5.5)[0], 2.5); // offset 0.5 -> t=1.5
        closeTo(evalTranslation(ch, 0)[0], 5); // pre side: offset -1 -> 3 -> mirrored 1
    });

    it("Linear extrapolates the edge slope (both sides)", () => {
        const ch = mk(1, 1); // Behavior.Linear
        closeTo(evalTranslation(ch, 0)[0], -5, 1e-3);
        closeTo(evalTranslation(ch, 4)[0], 15, 1e-3);
        closeTo(evalTranslation(ch, 5)[0], 20, 1e-3);
    });

    it("Linear extrapolates rotations via slerp", () => {
        // 90deg around z over t=1..3; at t=4 Linear extrapolation reaches 135deg.
        const s = Math.SQRT1_2;
        const ch: AnimationChannel = {
            nodeID: 0, path: "rotation", interp: "LINEAR",
            times: new Float32Array([1, 3]),
            values: new Float32Array([0, 0, 0, 1, 0, 0, s, s]),
            postInfinity: 1,
        };
        const nodes: SceneNode[] = [{ parent: -1, t: new float3(0, 0, 0), r: new quatf(0, 0, 0, 1), s: new float3(1, 1, 1) }];
        const anim: SceneAnimations = { nodes, channels: [ch], start: 0, duration: 1 };
        const g = evaluateGlobals(anim, 4)[0]!;
        // Loose tolerance: the epsilon-segment slope quantizes at f32 keyframe
        // precision (native extrapolates from the same float keys).
        closeTo(g.get(0, 0), Math.cos((135 * Math.PI) / 180), 0.05);
        closeTo(g.get(1, 0), Math.sin((135 * Math.PI) / 180), 0.05);
    });
});
