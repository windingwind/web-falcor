/**
 * Uniformly sampled spectrum mirroring Utils/Color/SampledSpectrum.h
 * (float or float3 samples; eval in float32 like native).
 */

import { AssertionError, RuntimeError } from "../../Core/Error.js";
import { float2, float3 } from "../Math/Vector.js";

const f32 = Math.fround;

/** Mirrors SpectrumInterpolation. */
export enum SpectrumInterpolation {
    Linear, ///< Piecewise linear between the two nearest samples; zero outside the end points.
}

export type SpectrumValue = number | float3;

/** Value type of a SampledSpectrum (native template parameter T). */
export type SpectrumValueType = "float" | "float3";

function toF32<T extends SpectrumValue>(v: T): T {
    return (typeof v === "number" ? f32(v) : new float3(f32(v.x), f32(v.y), f32(v.z))) as T;
}

/** Native float math::lerp: (1 - s) * x + s * y, per component. */
function lerpF32(x: number, y: number, s: number): number {
    return f32(f32(f32(1 - s) * x) + f32(s * y));
}

/**
 * Mirrors SampledSpectrum<T>: the first sample sits at lambdaStart, the last at
 * lambdaEnd, and the spectrum is zero outside that range.
 */
export class SampledSpectrum<T extends SpectrumValue = number> {
    private readonly lambdaStart: number;
    private readonly lambdaEnd: number;
    private samples: T[];
    readonly valueType: SpectrumValueType;

    /** Zero-initialized (native 3-arg ctor), or from `samples`; valueType defaults to the samples' type, else "float". */
    constructor(lambdaStart: number, lambdaEnd: number, sampleCount: number, samples?: ArrayLike<T> | null, valueType?: SpectrumValueType) {
        this.lambdaStart = f32(lambdaStart);
        this.lambdaEnd = f32(lambdaEnd);
        if (!(this.lambdaEnd > this.lambdaStart)) throw new RuntimeError("'lambdaEnd' must be larger than 'lambdaStart'.");
        if (!(sampleCount > 0)) throw new RuntimeError("'sampleCount' must be at least one.");
        this.valueType = valueType ?? (samples && samples.length > 0 && typeof samples[0] !== "number" ? "float3" : "float");
        this.samples = Array.from({ length: sampleCount }, () => this.zero());
        if (samples) this.set(samples);
    }

    /** value_type(0). */
    zero(): T {
        return (this.valueType === "float3" ? new float3(0, 0, 0) : 0) as T;
    }

    /** Mirrors set(samples) / set(index, value) / set(samples, lambdas). */
    set(samples: ArrayLike<T>): void;
    set(index: number, value: T): void;
    set(samples: ArrayLike<T>, lambdas: ArrayLike<number>): void;
    set(a: ArrayLike<T> | number, b?: T | ArrayLike<number>): void {
        if (typeof a === "number") {
            if (!(a >= 0 && a < this.size())) throw new AssertionError("SampledSpectrum.set: index out of range");
            this.samples[a] = toF32(b as T);
            return;
        }
        if (b !== undefined) {
            const lambdas = b as ArrayLike<number>;
            if (a.length === 0 || a.length !== lambdas.length) throw new RuntimeError("'samples' and 'lambdas' must be non-empty and of equal length.");
            throw new RuntimeError("Not implemented."); // native FALCOR_UNIMPLEMENTED
        }
        if (a.length !== this.samples.length) throw new RuntimeError("Sample count mismatch.");
        this.samples = Array.from(a, (v) => toF32(v));
    }

    /** Mirrors eval (float32 arithmetic). */
    eval(lambda: number, interpolationType: SpectrumInterpolation = SpectrumInterpolation.Linear): T {
        if (interpolationType !== SpectrumInterpolation.Linear) throw new RuntimeError("Interpolation type must be 'Linear'");
        lambda = f32(lambda);
        if (lambda < this.lambdaStart || lambda > this.lambdaEnd) return this.zero();
        const x = f32(f32(f32(lambda - this.lambdaStart) / f32(this.lambdaEnd - this.lambdaStart)) * f32(this.size() - 1));
        const i = Math.floor(x);
        if (i + 1 >= this.samples.length) return toF32(this.samples[this.size() - 1]!);
        const w = f32(x - i);
        const a = this.samples[i]!;
        const b = this.samples[i + 1]!;
        if (typeof a === "number") return lerpF32(a, b as number, w) as T;
        const bb = b as float3;
        return new float3(lerpF32(a.x, bb.x, w), lerpF32(a.y, bb.y, w), lerpF32(a.z, bb.z, w)) as T;
    }

    /** Mirrors toXYZ_CIE1931, which native leaves unimplemented (use SpectrumUtils.toXYZ). */
    toXYZ_CIE1931(): float3 {
        throw new RuntimeError("Not implemented.");
    }

    size(): number {
        return this.samples.length;
    }

    get(index: number): T {
        if (!(index >= 0 && index < this.size())) throw new AssertionError("SampledSpectrum.get: index out of range");
        return toF32(this.samples[index]!);
    }

    getWavelengthRange(): float2 {
        return new float2(this.lambdaStart, this.lambdaEnd);
    }
}
