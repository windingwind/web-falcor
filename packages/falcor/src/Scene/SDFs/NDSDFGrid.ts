/**
 * Normalized dense SDF grid host mirroring Scene/SDFs/NormalizedDenseSDFGrid/
 * NDSDFGrid.cpp plus the procedural generators from SDFs/SDFGrid.cpp.
 * All math is float32-exact vs the gcc-built native (Math.fround discipline;
 * mt19937 + generate_canonical<float,24> for generateCheeseValues — pinned
 * against tests/oracle/assets/ndsdf-cheese-ref.json).
 *
 * Web divergence (documented): the per-LOD R8Snorm 3D textures upload into
 * ONE atlas stacked along Z (WGSL has no binding arrays; the NDSDFGrid.slang
 * override consumes it at zBase(lod) = lod + c*(2^lod - 1)).
 */

import { Mt19937, canonicalFloat } from "../../Utils/SampleGenerators/CPUSampleGenerator.js";

const f = Math.fround;
const kSqrt3 = Math.fround(Math.sqrt(3));

function bitScanReverse(v: number): number {
    return 31 - Math.clz32(v);
}

/**
 * Mirrors SDFGrid::generateCheeseValues corner-value generation (shared by
 * all grid types; quantization differs per type). Bit-exact vs gcc — see
 * tests/oracle/ndsdf-cheese-ref.cpp.
 */
export function generateCheeseCornerValues(gridWidth: number, seed: number): Float32Array {
    const kHalfCheeseExtent = 0.4;
    const kHoleCount = 32;
    const holes = new Float32Array(kHoleCount * 4);

    const rng = new Mt19937(seed);
    // gcc evaluates float3(dist, dist, dist) constructor args RIGHT-TO-LEFT:
    // x receives the 3rd draw, y the 2nd, z the 1st (pinned by the C++ ref).
    for (let s = 0; s < kHoleCount; s++) {
        const d1 = canonicalFloat(rng);
        const d2 = canonicalFloat(rng);
        const d3 = canonicalFloat(rng);
        holes[s * 4] = f(f(2 * kHalfCheeseExtent * d3) - kHalfCheeseExtent);
        holes[s * 4 + 1] = f(f(2 * kHalfCheeseExtent * d2) - kHalfCheeseExtent);
        holes[s * 4 + 2] = f(f(2 * kHalfCheeseExtent * d1) - kHalfCheeseExtent);
        holes[s * 4 + 3] = f(f(canonicalFloat(rng) * 0.2) + 0.01);
    }

    const gridWidthInValues = gridWidth + 1;
    const cornerValues = new Float32Array(gridWidthInValues ** 3);
    const len3 = (x: number, y: number, z: number) => f(Math.sqrt(f(f(f(x * x) + f(y * y)) + f(z * z))));

    for (let z = 0; z < gridWidthInValues; z++) {
        for (let y = 0; y < gridWidthInValues; y++) {
            for (let x = 0; x < gridWidthInValues; x++) {
                const plx = f(f(x / gridWidth) - 0.5);
                const ply = f(f(y / gridWidth) - 0.5);
                const plz = f(f(z / gridWidth) - 0.5);

                // Box.
                const dx = f(Math.abs(plx) - kHalfCheeseExtent);
                const dy = f(Math.abs(ply) - kHalfCheeseExtent);
                const dz = f(Math.abs(plz) - kHalfCheeseExtent);
                const outsideDist = len3(Math.max(dx, 0), Math.max(dy, 0), Math.max(dz, 0));
                const insideDist = Math.min(Math.max(Math.max(dx, dy), dz), 0);
                let sd = f(outsideDist + insideDist);

                // Holes.
                for (let s = 0; s < kHoleCount; s++) {
                    const hd = f(len3(f(plx - holes[s * 4]!), f(ply - holes[s * 4 + 1]!), f(plz - holes[s * 4 + 2]!)) - holes[s * 4 + 3]!);
                    sd = Math.max(sd, -hd);
                }

                cornerValues[x + gridWidthInValues * (y + gridWidthInValues * z)] = Math.min(Math.max(sd, -kSqrt3), kSqrt3);
            }
        }
    }
    return cornerValues;
}

export class NDSDFGrid {
    readonly narrowBandThickness: number;
    gridWidth = 0;
    /** Per-LOD snorm8-quantized values, coarsest first (widths 1+(c<<lod)). */
    values: Int8Array[] = [];
    coarsestLODGridWidth = 0;
    coarsestLODNormalizationFactor = 0;

    constructor(narrowBandThickness: number) {
        this.narrowBandThickness = Math.max(f(narrowBandThickness), 1.0);
    }

    get lodCount(): number {
        return this.values.length;
    }

    get coarsestLODAsLevel(): number {
        return bitScanReverse(this.coarsestLODGridWidth);
    }

    /** Mirrors SDFGrid::generateCheeseValues. */
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.setValues(generateCheeseCornerValues(gridWidth, seed), gridWidth);
    }

    /** Mirrors NDSDFGrid::setValuesInternal (LOD chain + snorm8 quantization). */
    setValues(cornerValues: Float32Array, gridWidth: number): void {
        const kCoarsestAllowedGridWidth = 8;
        if (kCoarsestAllowedGridWidth > gridWidth) throw new Error(`NDSDFGrid: grid width must be larger than ${kCoarsestAllowedGridWidth}`);
        this.gridWidth = gridWidth;

        const lodCount = bitScanReverse(gridWidth / kCoarsestAllowedGridWidth) + 1;
        this.coarsestLODGridWidth = gridWidth >> (lodCount - 1);
        this.coarsestLODNormalizationFactor = this.calculateNormalizationFactor(this.coarsestLODGridWidth);

        this.values = [];
        const gridWidthInValues = gridWidth + 1;

        for (let lod = 0; lod < lodCount; lod++) {
            const lodWidthInValues = 1 + (this.coarsestLODGridWidth << lod);
            const normalizationFactor = this.coarsestLODNormalizationFactor / (1 << lod); // exact power-of-2 scale
            const lodValues = new Int8Array(lodWidthInValues ** 3);
            const lodReadStride = 1 << (lodCount - lod - 1);

            for (let z = 0; z < lodWidthInValues; z++) {
                for (let y = 0; y < lodWidthInValues; y++) {
                    for (let x = 0; x < lodWidthInValues; x++) {
                        const writeLocation = x + lodWidthInValues * (y + lodWidthInValues * z);
                        const readLocation = lodReadStride * (x + gridWidthInValues * (y + gridWidthInValues * z));
                        const normalizedValue = Math.min(Math.max(f(cornerValues[readLocation]! / normalizationFactor), -1), 1);
                        const integerScale = f(normalizedValue * 127);
                        // int8_t(x + 0.5f) truncates toward zero.
                        lodValues[writeLocation] = Math.trunc(integerScale >= 0 ? f(integerScale + 0.5) : f(integerScale - 0.5));
                    }
                }
            }
            this.values.push(lodValues);
        }
    }

    private calculateNormalizationFactor(gridWidth: number): number {
        return f(f(f(0.5 * kSqrt3) * this.narrowBandThickness) / gridWidth);
    }

    /**
     * Assembles the single-texture LOD atlas consumed by the NDSDFGrid.slang
     * override: XY = finest LOD width, LOD slabs stacked along Z at
     * zBase(lod) = lod + c*(2^lod - 1).
     */
    buildAtlas(): { data: Int8Array; width: number; height: number; depth: number } {
        const n = this.lodCount;
        const c = this.coarsestLODGridWidth;
        const width = 1 + (c << (n - 1));
        const depth = n + c * ((1 << n) - 1);
        const data = new Int8Array(width * width * depth);

        for (let lod = 0; lod < n; lod++) {
            const lw = 1 + (c << lod);
            const zBase = lod + c * ((1 << lod) - 1);
            const lodValues = this.values[lod]!;
            for (let z = 0; z < lw; z++) {
                for (let y = 0; y < lw; y++) {
                    data.set(lodValues.subarray(lw * (y + lw * z), lw * (y + lw * z) + lw), width * (y + width * (zBase + z)));
                }
            }
        }
        return { data, width, height: width, depth };
    }
}
