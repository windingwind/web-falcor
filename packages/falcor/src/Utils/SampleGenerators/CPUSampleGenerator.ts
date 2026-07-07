/**
 * CPU sample pattern generators mirroring Falcor/Utils/SampleGenerators/*.
 * Used for camera jitter (GBuffer/TAA). Values are pixel offsets in [-0.5, 0.5).
 */

import { float2, frac } from "../Math/Vector.js";
import { Logger } from "../Logger.js";

export interface CPUSampleGenerator {
    getSampleCount(): number;
    reset(startID?: number): void;
    next(): float2;
}

/** Mirrors DxSamplePattern: the D3D 8x MSAA pattern. */
export class DxSamplePattern implements CPUSampleGenerator {
    private static readonly kPattern: [number, number][] = [
        [1 / 16, -3 / 16],
        [-1 / 16, 3 / 16],
        [5 / 16, 1 / 16],
        [-3 / 16, -5 / 16],
        [-5 / 16, 5 / 16],
        [-7 / 16, -1 / 16],
        [3 / 16, 7 / 16],
        [7 / 16, -7 / 16],
    ];
    private curSample = 0;

    constructor(sampleCount = 8) {
        if (sampleCount !== 8) Logger.warning("DxSamplePattern() requires sampleCount = 8. Using eight samples.");
    }

    getSampleCount(): number { return 8; }
    reset(): void { this.curSample = 0; }
    next(): float2 {
        const p = DxSamplePattern.kPattern[this.curSample++ % 8]!;
        return new float2(p[0], p[1]);
    }
}

function halton(index: number, base: number): number {
    let result = 0;
    let factor = 1;
    for (; index > 0; index = Math.floor(index / base)) {
        factor /= base;
        result += factor * (index % base);
    }
    return result;
}

/** Mirrors HaltonSamplePattern. */
export class HaltonSamplePattern implements CPUSampleGenerator {
    private curSample = 0;

    constructor(private readonly sampleCount: number) {}

    getSampleCount(): number { return this.sampleCount; }
    reset(): void { this.curSample = 0; }
    next(): float2 {
        const value = new float2(halton(this.curSample, 2), halton(this.curSample, 3));
        this.curSample++;
        if (this.sampleCount !== 0) this.curSample %= this.sampleCount;
        // Map [0,1) to [-0.5, 0.5) with 0 at the origin.
        return new float2(frac(value.x + 0.5) - 0.5, frac(value.y + 0.5) - 0.5);
    }
}

/** std::mt19937 (32-bit Mersenne Twister, default seed 5489). */
export class Mt19937 {
    private mt = new Uint32Array(624);
    private index = 625;

    constructor(seed = 5489) {
        this.mt[0] = seed >>> 0;
        for (let i = 1; i < 624; i++) {
            const prev = this.mt[i - 1]! ^ (this.mt[i - 1]! >>> 30);
            const lo = (prev & 0xffff) * 1812433253;
            const hi = (((prev >>> 16) * 1812433253) & 0xffff) << 16;
            this.mt[i] = (((lo + hi) >>> 0) + i) >>> 0;
        }
        this.index = 624;
    }

    next(): number {
        if (this.index >= 624) {
            for (let i = 0; i < 624; i++) {
                const y = ((this.mt[i]! & 0x80000000) | (this.mt[(i + 1) % 624]! & 0x7fffffff)) >>> 0;
                let next = (this.mt[(i + 397) % 624]! ^ (y >>> 1)) >>> 0;
                if (y & 1) next = (next ^ 0x9908b0df) >>> 0;
                this.mt[i] = next;
            }
            this.index = 0;
        }
        let y = this.mt[this.index++]!;
        y = (y ^ (y >>> 11)) >>> 0;
        y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
        y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
        return (y ^ (y >>> 18)) >>> 0;
    }
}

/**
 * libstdc++ std::uniform_int_distribution over [0, b] driven by mt19937:
 * the downscaling rejection loop (bits/uniform_int_dist.h). All quantities
 * stay below 2^53.
 */
function uniformInt(rng: Mt19937, b: number): number {
    const urngrange = 4294967295;
    if (urngrange > b) {
        const uerange = b + 1;
        const scaling = Math.floor(urngrange / uerange);
        const past = uerange * scaling;
        let ret;
        do {
            ret = rng.next();
        } while (ret >= past);
        return Math.floor(ret / scaling);
    }
    return rng.next();
}

/** libstdc++ std::generate_canonical<float, 24, mt19937>: one draw, float math. */
export function canonicalFloat(rng: Mt19937): number {
    let ret = Math.fround(Math.fround(rng.next()) / 4294967296);
    if (ret >= 1) ret = 0.99999994; // nextafter(1.f, 0.f)
    return ret;
}

/**
 * libstdc++ std::shuffle (bits/stl_algo.h): mt19937's range covers n*n for all
 * our sample counts (n <= 1024), so the pairwise __gen_two_uniform_ints path
 * runs: odd leading swap for even n, then two swaps per distribution draw.
 */
function stdShuffle(arr: number[], rng: Mt19937): void {
    const n = arr.length;
    if (n === 0) return;
    const swap = (a: number, b: number) => {
        const t = arr[a]!;
        arr[a] = arr[b]!;
        arr[b] = t;
    };
    let i = 1;
    if (n % 2 === 0) {
        swap(i, uniformInt(rng, 1));
        i++;
    }
    while (i < n) {
        const swapRange = i + 1;
        // __gen_two_uniform_ints(swapRange, swapRange + 1, g)
        const x = uniformInt(rng, swapRange * (swapRange + 1) - 1);
        swap(i++, Math.floor(x / (swapRange + 1)));
        swap(i++, x % (swapRange + 1));
    }
}

/**
 * Mirrors StratifiedSamplePattern (MxN grid with per-round shuffled
 * permutation). Bit-exact vs native: replicates std::mt19937 plus the
 * libstdc++ implementations of std::shuffle, std::uniform_int_distribution
 * and std::generate_canonical<float, 24> used by the gcc-built oracles.
 */
export class StratifiedSamplePattern implements CPUSampleGenerator {
    private binsX: number;
    private binsY: number;
    private permutation: number[];
    private curSample = 0;
    private rng = new Mt19937();

    constructor(sampleCount = 1) {
        if (sampleCount < 1) Logger.warning("StratifiedSamplePattern() requires sampleCount > 0. Using one sample.");
        else if (sampleCount > 1024) Logger.warning("StratifiedSamplePattern() requires sampleCount <= 1024. Using 1024 samples.");
        sampleCount = Math.min(Math.max(sampleCount, 1), 1024);
        this.binsX = Math.floor(Math.sqrt(sampleCount));
        this.binsY = Math.floor(sampleCount / this.binsX);
        while (this.binsX * this.binsY !== sampleCount) {
            this.binsX++;
            this.binsY = Math.floor(sampleCount / this.binsX);
        }
        this.permutation = Array.from({ length: sampleCount }, (_v, i) => i);
    }

    getSampleCount(): number { return this.binsX * this.binsY; }
    reset(): void {
        this.curSample = 0;
        this.rng = new Mt19937();
    }
    next(): float2 {
        if (this.curSample === 0) stdShuffle(this.permutation, this.rng);
        const binIdx = this.permutation[this.curSample]!;
        const i = binIdx % this.binsX;
        const j = Math.floor(binIdx / this.binsX);
        this.curSample = (this.curSample + 1) % this.getSampleCount();
        const x = Math.fround(Math.fround(i + canonicalFloat(this.rng)) / this.binsX);
        const y = Math.fround(Math.fround(j + canonicalFloat(this.rng)) / this.binsY);
        return new float2(Math.fround(x - 0.5), Math.fround(y - 0.5));
    }
}
