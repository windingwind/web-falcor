/**
 * Scene animation interpolation: LINEAR, STEP, and glTF CUBICSPLINE (Hermite)
 * keyframe sampling through the public evaluateGlobals() API.
 */

import { describe, it, expect } from "vitest";
import { evaluateGlobals, type AnimationChannel, type SceneNode, type SceneAnimations } from "../src/Scene/Animation/SceneAnimation.js";
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
