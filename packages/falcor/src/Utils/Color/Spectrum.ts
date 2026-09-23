/**
 * Spectra mirroring Utils/Color/Spectrum.{h,cpp}: piecewise-linear, densely
 * sampled and blackbody spectra, the named spectra table (generated from
 * native's Spectra.inl, see scripts/gen-spectra.mjs), and the conversion to RGB
 * through the CIE 1931 curves and the Rec.709 matrix.
 *
 * Arithmetic follows native's float code: the inner product walks 1 nm steps
 * over the overlap of both ranges and sums in float32.
 */

import { RuntimeError } from "../../Core/Error.js";
import { float3 } from "../Math/Vector.js";
import { kCIE_X, kCIE_Y, kCIE_Z, kNamedSpectrumTables } from "./SpectraData.js";

const f32 = Math.fround;

interface SampledSpectrum {
    eval(wavelength: number): number;
    getWavelengthRange(): [number, number];
    getMaxValue(): number;
}

/** Mirrors PiecewiseLinearSpectrum. */
export class PiecewiseLinearSpectrum implements SampledSpectrum {
    private wavelengths: Float32Array;
    private values: Float32Array;
    private maxValue: number;

    constructor(wavelengths: ArrayLike<number>, values: ArrayLike<number>) {
        if (wavelengths.length !== values.length) throw new RuntimeError("'wavelengths' and 'values' need to contain the same number of elements");
        this.wavelengths = Float32Array.from(wavelengths);
        this.values = Float32Array.from(values);
        this.maxValue = Math.max(...this.values);
    }

    /** Mirrors fromInterleaved: (wavelength, value) pairs, optionally scaled to unit CIE Y. */
    static fromInterleaved(interleaved: ArrayLike<number>, normalize: boolean): PiecewiseLinearSpectrum {
        if (interleaved.length % 2 !== 0) throw new RuntimeError("'interleaved' must have an even number of elements.");
        const count = interleaved.length / 2;
        const wavelengths = new Float32Array(count);
        const values = new Float32Array(count);
        for (let i = 0; i < count; i++) {
            wavelengths[i] = interleaved[i * 2]!;
            values[i] = interleaved[i * 2 + 1]!;
            if (i > 0 && wavelengths[i]! < wavelengths[i - 1]!) throw new RuntimeError("'interleaved' must have wavelengths that are monotonic increasing.");
        }
        const spectrum = new PiecewiseLinearSpectrum(wavelengths, values);
        if (normalize) spectrum.scale(f32(kCIE_Y_Integral / innerProduct(spectrum, Spectra.kCIE_Y)));
        return spectrum;
    }

    scale(factor: number): void {
        if (!(factor >= 0)) throw new RuntimeError(`'factor' (${factor}) needs to be positive.`);
        for (let i = 0; i < this.values.length; i++) this.values[i] = f32(this.values[i]! * factor);
        this.maxValue = f32(this.maxValue * factor);
    }

    eval(wavelength: number): number {
        const w = this.wavelengths;
        if (w.length === 0 || wavelength < w[0]! || wavelength > w[w.length - 1]!) return 0;
        // std::lower_bound: first element not less than the wavelength.
        let lo = 0;
        let hi = w.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (w[mid]! < wavelength) lo = mid + 1;
            else hi = mid;
        }
        if (lo === 0) return this.values[0]!;
        const index = lo - 1;
        const t = f32((wavelength - w[index]!) / (w[index + 1]! - w[index]!));
        const a = this.values[index]!;
        const b = this.values[index + 1]!;
        return f32(a + (b - a) * t);
    }

    getWavelengthRange(): [number, number] {
        return [this.wavelengths[0]!, this.wavelengths[this.wavelengths.length - 1]!];
    }

    getMaxValue(): number {
        return this.maxValue;
    }
}

/** Mirrors DenseleySampledSpectrum (uniform samples over a range). */
export class DenselySampledSpectrum implements SampledSpectrum {
    private step: number;
    private maxValue: number;

    constructor(
        private readonly minWavelength: number,
        private readonly maxWavelength: number,
        private readonly values: Float32Array,
    ) {
        this.step = f32((maxWavelength - minWavelength) / (values.length - 1));
        this.maxValue = Math.max(...values);
    }

    eval(wavelength: number): number {
        // std::lroundf: round half away from zero.
        const x = (wavelength - this.minWavelength) / this.step;
        const index = x < 0 ? -Math.round(-x) : Math.round(x);
        if (index < 0 || index >= this.values.length) return 0;
        return this.values[index]!;
    }

    getWavelengthRange(): [number, number] {
        return [this.minWavelength, this.maxWavelength];
    }

    getMaxValue(): number {
        return this.maxValue;
    }
}

/** Mirrors blackbodyEmission: Planck's law, wavelength in nm, temperature in K. */
export function blackbodyEmission(wavelength: number, temperature: number): number {
    if (temperature <= 0) return 0;
    const c = 299792458;
    const h = 6.62606957e-34;
    const kb = 1.3806488e-23;
    const l = wavelength * 1e-9;
    return (2 * h * c * c) / (l ** 5 * (Math.exp((h * c) / (l * kb * temperature)) - 1));
}

/** Mirrors BlackbodySpectrum (normalized to a peak of 1 by default). */
export class BlackbodySpectrum implements SampledSpectrum {
    private normalization: number;
    private maxValue: number;

    constructor(
        readonly temperature: number,
        normalize = true,
    ) {
        // Wien's displacement law gives the peak wavelength.
        const peakWavelength = (2.8977721e-3 / temperature) * 1e9;
        const peakValue = blackbodyEmission(peakWavelength, temperature);
        this.normalization = normalize ? 1 / peakValue : 1;
        this.maxValue = normalize ? 1 : peakValue;
    }

    eval(wavelength: number): number {
        return blackbodyEmission(wavelength, this.temperature) * this.normalization;
    }

    getWavelengthRange(): [number, number] {
        return [-Infinity, Infinity];
    }

    getMaxValue(): number {
        return this.maxValue;
    }
}

/** Integral of CIE Y over its range (native Spectra::kCIE_Y_Integral). */
export const kCIE_Y_Integral = 106.856895;

/** Mirrors innerProduct: sum over 1 nm steps across the common range. */
export function innerProduct(a: SampledSpectrum, b: SampledSpectrum): number {
    const [a0, a1] = a.getWavelengthRange();
    const [b0, b1] = b.getWavelengthRange();
    const minWavelength = Math.max(a0, b0);
    const maxWavelength = Math.min(a1, b1);
    let integral = 0;
    for (let wavelength = minWavelength; wavelength <= maxWavelength; wavelength = f32(wavelength + 1)) {
        integral = f32(integral + f32(a.eval(wavelength) * b.eval(wavelength)));
    }
    return integral;
}

/** Mirrors Spectra: the CIE curves and the named spectra. */
export const Spectra = {
    kCIE_X: new DenselySampledSpectrum(360, 830, kCIE_X),
    kCIE_Y: new DenselySampledSpectrum(360, 830, kCIE_Y),
    kCIE_Z: new DenselySampledSpectrum(360, 830, kCIE_Z),
    /** Mirrors Spectra::getNamedSpectrum (undefined for unknown names). */
    getNamedSpectrum(name: string): PiecewiseLinearSpectrum | undefined {
        let spectrum = namedCache.get(name);
        if (!spectrum) {
            const table = kNamedSpectrumTables[name];
            if (!table) return undefined;
            spectrum = PiecewiseLinearSpectrum.fromInterleaved(table.interleaved, table.normalize);
            namedCache.set(name, spectrum);
        }
        return spectrum;
    },
};

const namedCache = new Map<string, PiecewiseLinearSpectrum>();

/** Mirrors spectrumToXYZ. */
export function spectrumToXYZ(s: SampledSpectrum): float3 {
    return new float3(
        innerProduct(s, Spectra.kCIE_X) / kCIE_Y_Integral,
        innerProduct(s, Spectra.kCIE_Y) / kCIE_Y_Integral,
        innerProduct(s, Spectra.kCIE_Z) / kCIE_Y_Integral,
    );
}

/** Mirrors ColorHelpers' XYZtoRGB_Rec709. */
export function XYZtoRGB_Rec709(c: float3): float3 {
    return new float3(
        3.240969941904522 * c.x - 1.537383177570094 * c.y - 0.4986107602930032 * c.z,
        -0.9692436362808803 * c.x + 1.875967501507721 * c.y + 0.04155505740717569 * c.z,
        0.05563007969699373 * c.x - 0.2039769588889765 * c.y + 1.056971514242878 * c.z,
    );
}

/** Mirrors spectrumToRGB. */
export function spectrumToRGB(s: SampledSpectrum): float3 {
    return XYZtoRGB_Rec709(spectrumToXYZ(s));
}
