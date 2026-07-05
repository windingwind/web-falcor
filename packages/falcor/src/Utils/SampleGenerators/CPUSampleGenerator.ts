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

/** Mirrors StratifiedSamplePattern (MxN grid with per-round shuffled permutation). */
export class StratifiedSamplePattern implements CPUSampleGenerator {
    private binsX: number;
    private binsY: number;
    private permutation: number[];
    private curSample = 0;
    private rngState: number;

    constructor(sampleCount = 1, seed = 5489 /* mt19937 default_seed */) {
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
        this.rngState = seed >>> 0;
    }

    // Divergence note: upstream uses std::mt19937; we use xorshift32. The pattern
    // is stochastic by design — only distribution properties matter.
    private rand(): number {
        let x = this.rngState;
        x ^= x << 13; x >>>= 0;
        x ^= x >> 17;
        x ^= x << 5; x >>>= 0;
        this.rngState = x;
        return x / 0x100000000;
    }

    getSampleCount(): number { return this.binsX * this.binsY; }
    reset(): void { this.curSample = 0; }
    next(): float2 {
        if (this.curSample === 0) {
            for (let i = this.permutation.length - 1; i > 0; i--) {
                const j = Math.floor(this.rand() * (i + 1));
                [this.permutation[i], this.permutation[j]] = [this.permutation[j]!, this.permutation[i]!];
            }
        }
        const binIdx = this.permutation[this.curSample]!;
        const i = binIdx % this.binsX;
        const j = Math.floor(binIdx / this.binsX);
        this.curSample = (this.curSample + 1) % this.getSampleCount();
        const x = (i + this.rand()) / this.binsX;
        const y = (j + this.rand()) / this.binsY;
        return new float2(x - 0.5, y - 0.5);
    }
}
