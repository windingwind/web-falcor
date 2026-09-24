/**
 * Port of Falcor/Scene/Animation/Animation (the python `Animation` class): transform keyframes on
 * one scene-graph node, interpolated linearly (lerp translation/scale, slerp rotation) or with the
 * Bezier-form Hermite spline, with optional warping (the last keyframe blends into the first) and
 * pre/post-infinity behaviors. animate() gives the node's local translation/rotation/scaling.
 */

import { float3 } from "../../Utils/Math/Vector.js";
import { quatf, slerp } from "../../Utils/Math/Quaternion.js";
import { AnimationBehavior } from "./SceneAnimation.js";

export enum InterpolationMode {
    Linear = 0,
    Hermite = 1,
}

export interface Keyframe {
    time: number;
    translation: float3;
    scaling: float3;
    rotation: quatf;
}

const kEpsilonTime = 1e-5;
const lerp3 = (a: float3, b: float3, t: number) => new float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);

/** Bezier-form Hermite (Catmull-Rom tangents) on float3. */
function hermite3(p0: float3, p1: float3, p2: float3, p3: float3, t: number): float3 {
    const k = 0.5 / 3;
    const b1 = new float3(p1.x + (p2.x - p0.x) * k, p1.y + (p2.y - p0.y) * k, p1.z + (p2.z - p0.z) * k);
    const b2 = new float3(p2.x - (p3.x - p1.x) * k, p2.y - (p3.y - p1.y) * k, p2.z - (p3.z - p1.z) * k);
    const [q0, q1, q2] = [lerp3(p1, b1, t), lerp3(b1, b2, t), lerp3(b2, p2, t)];
    return lerp3(lerp3(q0, q1, t), lerp3(q1, q2, t), t);
}

/** The same spline with slerp between the (unnormalized) control quaternions, as natively. */
function hermiteQuat(r0: quatf, r1: quatf, r2: quatf, r3: quatf, t: number): quatf {
    const k = 0.5 / 3;
    const b1 = new quatf(r1.x + (r2.x - r0.x) * k, r1.y + (r2.y - r0.y) * k, r1.z + (r2.z - r0.z) * k, r1.w + (r2.w - r0.w) * k);
    const b2 = new quatf(r2.x - (r3.x - r1.x) * k, r2.y - (r3.y - r1.y) * k, r2.z - (r3.z - r1.z) * k, r2.w - (r3.w - r1.w) * k);
    const [q0, q1, q2] = [slerp(r1, b1, t), slerp(b1, b2, t), slerp(b2, r2, t)];
    return slerp(slerp(q0, q1, t), slerp(q1, q2, t), t);
}

function interpolateLinear(k0: Keyframe, k1: Keyframe, t: number): Keyframe {
    return {
        translation: lerp3(k0.translation, k1.translation, t),
        scaling: lerp3(k0.scaling, k1.scaling, t),
        rotation: slerp(k0.rotation, k1.rotation, t),
        time: k0.time + (k1.time - k0.time) * t,
    };
}

export class KeyframeAnimation {
    interpolationMode = InterpolationMode.Linear;
    preInfinityBehavior = AnimationBehavior.Constant;
    postInfinityBehavior = AnimationBehavior.Constant;
    enableWarping = false;
    private readonly keyframes: Keyframe[] = [];

    constructor(
        readonly name: string,
        public nodeID: number,
        readonly duration: number,
    ) {}

    getKeyframes(): readonly Keyframe[] {
        return this.keyframes;
    }

    /** Mirrors Animation::addKeyframe: keeps time order and replaces a keyframe at the same time. */
    addKeyframe(keyframe: Keyframe): void {
        const i = this.keyframes.findIndex((k) => k.time >= keyframe.time);
        if (i < 0) this.keyframes.push(keyframe);
        else if (this.keyframes[i]!.time === keyframe.time) this.keyframes[i] = keyframe;
        else this.keyframes.splice(i, 0, keyframe);
    }

    /** Mirrors Animation::animate: the node's local transform at `currentTime`. */
    animate(currentTime: number): Keyframe {
        const first = this.keyframes[0]!;
        const last = this.keyframes[this.keyframes.length - 1]!;
        let time = currentTime;
        if (time < first.time || time > last.time) time = this.calcSampleTime(currentTime);
        const linearPost = time > last.time && this.postInfinityBehavior === AnimationBehavior.Linear;
        const linearPre = time < first.time && this.preInfinityBehavior === AnimationBehavior.Linear;
        if (linearPre && this.keyframes.length > 1) {
            const k1 = this.interpolate(this.interpolationMode, first.time + kEpsilonTime);
            return interpolateLinear(first, k1, Math.fround((time - first.time) / (k1.time - first.time)));
        }
        if (linearPost && this.keyframes.length > 1) {
            const k0 = this.interpolate(this.interpolationMode, last.time - kEpsilonTime);
            return interpolateLinear(k0, last, Math.fround((time - k0.time) / (last.time - k0.time)));
        }
        return this.interpolate(this.interpolationMode, time);
    }

    private interpolate(mode: InterpolationMode, time: number): Keyframe {
        const n = this.keyframes.length;
        let frame = 0;
        while (frame < n - 1 && !(this.keyframes[frame + 1]!.time > time)) frame++;
        const adjacent = (f: number, offset = 1) => (this.enableWarping ? (f + n + offset) % n : Math.min(Math.max(f + offset, 0), n - 1));
        const segment = (a: Keyframe, b: Keyframe) => {
            let d = b.time - a.time;
            if (this.enableWarping && d < 0) d += this.duration;
            return Math.fround(Math.min(Math.max(d > 0 ? (time - a.time) / d : 1, 0), 1));
        };
        if (mode === InterpolationMode.Linear || n < 4) {
            const [k0, k1] = [this.keyframes[frame]!, this.keyframes[adjacent(frame)]!];
            return interpolateLinear(k0, k1, segment(k0, k1));
        }
        const [k0, k1, k2, k3] = [this.keyframes[adjacent(frame, -1)]!, this.keyframes[frame]!, this.keyframes[adjacent(frame, 1)]!, this.keyframes[adjacent(frame, 2)]!];
        const t = segment(k1, k2);
        return {
            translation: hermite3(k0.translation, k1.translation, k2.translation, k3.translation, t),
            scaling: lerp3(k1.scaling, k2.scaling, t),
            rotation: hermiteQuat(k0.rotation, k1.rotation, k2.rotation, k3.rotation, t),
            time: k1.time + (k2.time - k1.time) * t,
        };
    }

    /** Mirrors Animation::calcSampleTime (outside the keyframe range). */
    private calcSampleTime(currentTime: number): number {
        const first = this.keyframes[0]!.time;
        const last = this.keyframes[this.keyframes.length - 1]!.time;
        const duration = last - first;
        const behavior = currentTime < first ? this.preInfinityBehavior : this.postInfinityBehavior;
        switch (behavior) {
            case AnimationBehavior.Cycle: {
                let t = first + ((currentTime - first) % duration);
                if (t < first) t += duration;
                return t;
            }
            case AnimationBehavior.Oscillate: {
                let offset = (currentTime - first) % (2 * duration);
                if (offset < 0) offset += 2 * duration;
                if (offset > duration) offset = 2 * duration - offset;
                return first + offset;
            }
            case AnimationBehavior.Linear:
                return currentTime;
            default:
                return Math.min(Math.max(currentTime, first), last);
        }
    }
}
