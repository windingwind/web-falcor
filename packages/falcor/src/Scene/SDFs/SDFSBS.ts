/**
 * Sparse-brick-set SDF grid host mirroring Scene/SDFs/SparseBrickSet/
 * SDFSBS.cpp createResourcesFromSDField. The native build runs five compute
 * kernels (assign brick validity, prefix-sum compaction, reset validity, copy
 * indirection, create bricks + AABBs); this ports them to the CPU so the whole
 * chain is deterministic in the browser (mirrors the NanoVDB CPU-build
 * precedent, docs §6.3). Uncompressed bricks only (createSBS default;
 * BC4Snorm compression needs the GPU BC4 encoder — documented gap).
 *
 * The runtime shader (SDFSBS.slang) is consumed UNMODIFIED: the host produces
 * exactly the resources its kernels would have (bricks Texture2D<float>,
 * indirection Texture3D<uint>, brick AABB buffer + uniforms).
 */

import { generateCheeseCornerValues } from "./NDSDFGrid.js";

const kSqrt3 = Math.sqrt(3);
const UINT32_MAX = 0xffffffff;

function bitScanReverse(v: number): number {
    return 31 - Math.clz32(v);
}

/** One brick's world-space AABB (matches Utils/Math/AABB, 6 floats). */
export interface BrickAABB {
    min: [number, number, number];
    max: [number, number, number];
}

export class SDFSBS {
    readonly brickWidth: number;
    gridWidth = 0;
    /** snorm8 dense field, (gridWidth+1)^3, row-major x-fastest. */
    private sdField = new Int8Array(0);

    // Build outputs (populated by build()).
    virtualBricksPerAxis = 0;
    brickCount = 0;
    bricksPerAxis: [number, number] = [0, 0];
    brickTextureDimensions: [number, number] = [0, 0];
    /** R32Float brick texture data (row-major). */
    brickTexture = new Float32Array(0);
    /** R32Uint indirection texture, virtualBricksPerAxis^3 (row-major). */
    indirection = new Uint32Array(0);
    aabbs: BrickAABB[] = [];

    constructor(brickWidth = 7) {
        this.brickWidth = brickWidth;
    }

    get normalizationFactor(): number {
        return (0.5 * kSqrt3) / this.gridWidth;
    }

    get maxPrimitiveIDBits(): number {
        return bitScanReverse(Math.max(this.brickCount - 1, 1)) + 1;
    }

    /** Mirrors SDFGrid::generateCheeseValues + SDFSBS::setValuesInternal. */
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.setValues(generateCheeseCornerValues(gridWidth, seed), gridWidth);
        this.build();
    }

    /** Mirrors SDFSBS::setValuesInternal (snorm8, normalizationFactor 2*gw/sqrt3). */
    setValues(cornerValues: Float32Array, gridWidth: number): void {
        this.gridWidth = gridWidth;
        const gwv = gridWidth + 1;
        this.sdField = new Int8Array(gwv * gwv * gwv);
        const normalizationFactor = (2 * gridWidth) / kSqrt3;
        for (let v = 0; v < this.sdField.length; v++) {
            const normalized = Math.min(Math.max(cornerValues[v]! * normalizationFactor, -1), 1);
            const integerScale = normalized * 127;
            this.sdField[v] = Math.trunc(integerScale >= 0 ? integerScale + 0.5 : integerScale - 0.5);
        }
    }

    private field(x: number, y: number, z: number): number {
        const gwv = this.gridWidth + 1;
        return this.sdField[x + gwv * (y + gwv * z)]! / 127;
    }

    /** Runs the five-kernel brick build on the CPU. */
    private build(): void {
        const gw = this.gridWidth;
        const bw = this.brickWidth;
        const bwv = bw + 1;
        const vbpa = Math.max(this.virtualBricksPerAxis, Math.ceil(gw / bw));
        this.virtualBricksPerAxis = vbpa;
        const virtualBrickCount = vbpa * vbpa * vbpa;

        // 1) Brick validity: a brick is valid if any of its voxels contains the
        //    surface (any corner <= 0 && any corner >= 0).
        const validity = new Uint8Array(virtualBrickCount);
        for (let vz = 0; vz < gw; vz++) {
            for (let vy = 0; vy < gw; vy++) {
                for (let vx = 0; vx < gw; vx++) {
                    // Voxel corner values.
                    let anyNeg = false;
                    let anyPos = false;
                    for (let cz = 0; cz <= 1; cz++)
                        for (let cy = 0; cy <= 1; cy++)
                            for (let cx = 0; cx <= 1; cx++) {
                                const val = this.field(vx + cx, vy + cy, vz + cz);
                                if (val <= 0) anyNeg = true;
                                if (val >= 0) anyPos = true;
                            }
                    if (anyNeg && anyPos) {
                        const bx = Math.floor(vx / bw);
                        const by = Math.floor(vy / bw);
                        const bz = Math.floor(vz / bw);
                        validity[bx + vbpa * (by + vbpa * bz)] = 1;
                    }
                }
            }
        }

        // 2) Prefix-sum compaction → brickID per valid brick; total brick count.
        const indirectionBuffer = new Uint32Array(virtualBrickCount);
        let running = 0;
        for (let i = 0; i < virtualBrickCount; i++) {
            indirectionBuffer[i] = running; // exclusive prefix sum
            running += validity[i]!;
        }
        this.brickCount = running;

        // 3) Reset invalid entries to UINT32_MAX.
        for (let i = 0; i < virtualBrickCount; i++) {
            if (validity[i] === 0) indirectionBuffer[i] = UINT32_MAX;
        }

        // 4) Indirection texture = indirectionBuffer laid out as vbpa^3 (R32Uint).
        this.indirection = indirectionBuffer;

        // 5) Brick texture layout (mirror the kernel's square-ish packing).
        const bricksAlongX = Math.max(1, Math.ceil(Math.sqrt(this.brickCount / bwv)));
        const bricksAlongY = Math.max(1, Math.ceil(this.brickCount / bricksAlongX));
        this.bricksPerAxis = [bricksAlongX, bricksAlongY];
        const textureWidth = bwv * bwv * bricksAlongX;
        const textureHeight = bwv * bricksAlongY;
        this.brickTextureDimensions = [textureWidth, textureHeight];
        this.brickTexture = new Float32Array(textureWidth * textureHeight);
        this.aabbs = new Array(this.brickCount);

        const oneOverGridWidth = 1 / gw;
        for (let vbz = 0; vbz < vbpa; vbz++) {
            for (let vby = 0; vby < vbpa; vby++) {
                for (let vbx = 0; vbx < vbpa; vbx++) {
                    const virtualBrickID = vbx + vbpa * (vby + vbpa * vbz);
                    const brickID = indirectionBuffer[virtualBrickID]!;
                    if (brickID === UINT32_MAX) continue;

                    // AABB (world-space, grid spans [-0.5, 0.5]).
                    const brickAABBMin: [number, number, number] = [
                        -0.5 + vbx * bw * oneOverGridWidth,
                        -0.5 + vby * bw * oneOverGridWidth,
                        -0.5 + vbz * bw * oneOverGridWidth,
                    ];
                    const brickAABBMax: [number, number, number] = [
                        Math.min(brickAABBMin[0] + bw * oneOverGridWidth, 0.5),
                        Math.min(brickAABBMin[1] + bw * oneOverGridWidth, 0.5),
                        Math.min(brickAABBMin[2] + bw * oneOverGridWidth, 0.5),
                    ];
                    this.aabbs[brickID] = { min: brickAABBMin, max: brickAABBMax };

                    // Brick texel base (matches calculateBrickTexelCoords).
                    const tcx = (brickID % bricksAlongX) * bwv * bwv;
                    const tcy = Math.floor(brickID / bricksAlongX) * bwv;
                    const gx = vbx * bw;
                    const gy = vby * bw;
                    const gz = vbz * bw;
                    for (let z = 0; z < bwv; z++) {
                        for (let y = 0; y < bwv; y++) {
                            for (let x = 0; x < bwv; x++) {
                                const wx = gx + x;
                                const wy = gy + y;
                                const wz = gz + z;
                                // Native: `all(voxelGridCoords < virtualGridWidth)` with
                                // virtualGridWidth == gridWidth (strict) — the last corner
                                // layer at index gridWidth is the 1.0 sentinel.
                                const value = wx < gw && wy < gw && wz < gw ? this.field(wx, wy, wz) : 1.0;
                                const px = tcx + x + z * bwv;
                                const py = tcy + y;
                                this.brickTexture[px + textureWidth * py] = value;
                            }
                        }
                    }
                }
            }
        }
    }
}
