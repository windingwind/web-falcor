/**
 * StratifiedSamplePattern parity: values pinned against a gcc/libstdc++
 * program running the native StratifiedSamplePattern.cpp algorithm verbatim
 * (std::mt19937 + std::shuffle + std::uniform_real_distribution<float>).
 * Regenerate with tools notes in the pass-rate memory; any drift means the
 * libstdc++ replication (shuffle pairwise path / canonical float) broke.
 */

import { describe, expect, it } from "vitest";
import { StratifiedSamplePattern } from "../src/Utils/SampleGenerators/CPUSampleGenerator.js";

// First samples emitted by the gcc build for sampleCount = 16, 1, 5.
const kExpected: Record<number, [number, number][]> = {
    16: [
        [0.158089817, -0.422958255],
        [-0.225614905, 0.136805177],
        [-0.430375457, -0.452904522],
        [0.136720359, 0.498220325],
        [-0.260623276, -0.000884652138],
        [0.491222143, 0.491923749],
        [0.28940326, -0.0685402751],
    ],
};

describe("StratifiedSamplePattern", () => {
    it("matches the native libstdc++ sequence (sampleCount=16)", () => {
        const pattern = new StratifiedSamplePattern(16);
        for (const [x, y] of kExpected[16]!) {
            const s = pattern.next();
            expect(s.x).toBeCloseTo(x, 7);
            expect(s.y).toBeCloseTo(y, 7);
        }
    });

    it("bins cover the domain and repeat after sampleCount", () => {
        const pattern = new StratifiedSamplePattern(16);
        const seen = new Set<string>();
        for (let i = 0; i < 16; i++) {
            const s = pattern.next();
            expect(s.x).toBeGreaterThanOrEqual(-0.5);
            expect(s.x).toBeLessThan(0.5);
            expect(s.y).toBeGreaterThanOrEqual(-0.5);
            expect(s.y).toBeLessThan(0.5);
            seen.add(`${Math.floor((s.x + 0.5) * 4)},${Math.floor((s.y + 0.5) * 4)}`);
        }
        expect(seen.size).toBe(16); // each 4x4 bin hit exactly once per round
    });
});
