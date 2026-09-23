/**
 * Spectrum helpers mirroring Utils/Color/SpectrumUtils.{h,cpp}: 1 nm CIE 1931
 * XYZ curves, 5 nm D65, and Riemann-sum integration of SampledSpectrum (float32).
 */

import { RuntimeError } from "../../Core/Error.js";
import { float3 } from "../Math/Vector.js";
import { SampledSpectrum, SpectrumInterpolation, type SpectrumValue } from "./SampledSpectrum.js";
import { kCIE_XYZ_1931_1nm, kNamedSpectrumTables } from "./SpectraData.js";

const f32 = Math.fround;

/** Native kColorTransform_XYZtoRGB_Rec709 (ColorUtils.h), row-major, as float. */
const kXYZtoRGB_Rec709 = new Float32Array([
    3.2409699419045213, -1.5373831775700935, -0.4986107602930033,
    -0.9692436362808798, 1.8759675015077206, 0.0415550574071756,
    0.0556300796969936, -0.2039769588889765, 1.0569715142428784,
]);

/** Native float mul(float3x3, float3): per-row float dot. */
function mulXYZtoRGB(c: float3): float3 {
    const m = kXYZtoRGB_Rec709;
    const row = (r: number) => f32(f32(f32(m[r * 3]! * c.x) + f32(m[r * 3 + 1]! * c.y)) + f32(m[r * 3 + 2]! * c.z));
    return new float3(row(0), row(1), row(2));
}

/** Native D65_1nm[107] (5 nm, 300..830); bit-identical to Spectra.inl's CIE_Illum_D6500 values. */
function d65Samples(): Float32Array {
    const interleaved = kNamedSpectrumTables["stdillum-D65"]!.interleaved;
    const out = new Float32Array(interleaved.length / 2);
    for (let i = 0; i < out.length; i++) {
        if (interleaved[2 * i] !== 300 + 5 * i) throw new RuntimeError("SpectrumUtils: unexpected D65 table layout");
        out[i] = interleaved[2 * i + 1]!;
    }
    return out;
}

function xyzSamples(): float3[] {
    const out: float3[] = [];
    for (let i = 0; i < kCIE_XYZ_1931_1nm.length; i += 3) out.push(new float3(kCIE_XYZ_1931_1nm[i]!, kCIE_XYZ_1931_1nm[i + 1]!, kCIE_XYZ_1931_1nm[i + 2]!));
    return out;
}

/** Mirrors SpectrumUtils (static members only). */
export class SpectrumUtils {
    static readonly sCIE_XYZ_1931_1nm = new SampledSpectrum<float3>(360, 830, 471, xyzSamples()); // 1 nm between samples.
    static readonly sD65_5nm = new SampledSpectrum<number>(300, 830, 107, Array.from(d65Samples())); // 5 nm between samples.

    /** 1931 CIE XYZ matching curves at `lambda` nm (linear between 1 nm samples). */
    static wavelengthToXYZ_CIE1931(lambda: number): float3 {
        return SpectrumUtils.sCIE_XYZ_1931_1nm.eval(lambda);
    }

    /** D65 standard illuminant at `lambda` nm (linear between 5 nm samples). */
    static wavelengthToD65(lambda: number): number {
        return SpectrumUtils.sD65_5nm.eval(lambda);
    }

    /** wavelengthToXYZ_CIE1931 followed by XYZtoRGB_Rec709. */
    static wavelengthToRGB_Rec709(lambda: number): float3 {
        return mulXYZtoRGB(SpectrumUtils.wavelengthToXYZ_CIE1931(lambda));
    }

    /** Trapezoidal Riemann sum of func(wavelength) * spectrum over the spectrum's range. */
    static integrate<T extends SpectrumValue, R extends SpectrumValue>(
        spectrum: SampledSpectrum<T>,
        interpolationType: SpectrumInterpolation,
        func: (wavelength: number) => R,
        componentIndex = 0,
        integrationSteps = 1,
    ): R {
        if (!(integrationSteps >= 1)) throw new RuntimeError("integrationSteps must be at least 1");
        const range = spectrum.getWavelengthRange();
        const numEvaluations = spectrum.size() + (integrationSteps - 1) * (spectrum.size() - 1);
        const delta = f32(f32(range.y - range.x) / f32(numEvaluations - 1));
        let sumS = 0;
        let sumV: float3 | null = null;
        for (let q = 0; q < numEvaluations; q++) {
            const wavelength = Math.min(f32(range.x + f32(delta * f32(q))), range.y);
            const intensity = spectrum.eval(wavelength, interpolationType);
            const s = typeof intensity === "number" ? intensity : [intensity.x, intensity.y, intensity.z][componentIndex]!;
            const weight = q === 0 || q === numEvaluations - 1 ? 0.5 : 1;
            const term = (v: number) => f32(f32(f32(v * s) * delta) * weight);
            const value = func(wavelength);
            if (typeof value === "number") {
                sumS = f32(sumS + term(value));
            } else {
                sumV ??= new float3(0, 0, 0);
                sumV = new float3(f32(sumV.x + term(value.x)), f32(sumV.y + term(value.y)), f32(sumV.z + term(value.z)));
            }
        }
        return (sumV ?? sumS) as R;
    }

    /** Converts the whole spectrum to CIE 1931 XYZ. */
    static toXYZ<T extends SpectrumValue>(spectrum: SampledSpectrum<T>, interpolationType = SpectrumInterpolation.Linear, componentIndex = 0, integrationSteps = 1): float3 {
        return SpectrumUtils.integrate(spectrum, interpolationType, (w) => SpectrumUtils.wavelengthToXYZ_CIE1931(w), componentIndex, integrationSteps);
    }

    /** Converts the whole spectrum to XYZ times D65. */
    static toXYZ_D65<T extends SpectrumValue>(spectrum: SampledSpectrum<T>, interpolationType = SpectrumInterpolation.Linear, componentIndex = 0, integrationSteps = 1): float3 {
        return SpectrumUtils.integrate(
            spectrum,
            interpolationType,
            (w) => {
                const xyz = SpectrumUtils.wavelengthToXYZ_CIE1931(w);
                const d = SpectrumUtils.wavelengthToD65(w);
                return new float3(f32(xyz.x * d), f32(xyz.y * d), f32(xyz.z * d));
            },
            componentIndex,
            integrationSteps,
        );
    }

    /** RGB under D65 (Eq. 8 of "An OpenEXR Layout for Spectral Images", JCGT 2021). */
    static toRGB_D65<T extends SpectrumValue>(spectrum: SampledSpectrum<T>, interpolationType: SpectrumInterpolation, componentIndex = 0, integrationSteps = 1): float3 {
        const rgb = mulXYZtoRGB(SpectrumUtils.toXYZ_D65(spectrum, interpolationType, componentIndex, integrationSteps));
        const invY = f32(1 / f32(10567.0762)); // Y_D65 = sD65_5nm.toXYZ(1).y
        return new float3(f32(rgb.x * invY), f32(rgb.y * invY), f32(rgb.z * invY));
    }
}
