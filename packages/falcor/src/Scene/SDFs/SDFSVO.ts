/**
 * Sparse-voxel-octree SDF grid host mirroring Scene/SDFs/SparseVoxelOctree/
 * SDFSVO.cpp + its build kernels. Phase 1: setValues + the 64-bit Morton
 * location-code primitives (SDFVoxelCommon.slang encodeLocation /
 * createChildLocationCode / decode), ported via BigInt (WGSL/JS have no
 * native u64). Phase 2 builds the octree bottom-up and packs the
 * SDFSVOVoxel buffer the unmodified runtime traverses.
 *
 * The SVO runtime is a whole-grid octree walk (single procedural primitive,
 * like NDSDF — no per-primitive AABB BVH needed).
 */

import { generateCheeseCornerValues } from "./NDSDFGrid.js";

const kSqrt3 = Math.sqrt(3);
const kMaxLevel = 19n;
const kLocationCodeLevelOffset = 3n * kMaxLevel; // 57
const kLocationCodeVoxelCoordsMask = (1n << (3n * kMaxLevel)) - 1n;

function bitScanReverse(v: number): number {
    return 31 - Math.clz32(v);
}

/** Morton bit-spread of a 21-bit coord into every 3rd bit (u64). */
export function shiftCoord(x: number): bigint {
    let y = BigInt(x >>> 0);
    y = (y | (y << 32n)) & 0x1f00000000ffffn;
    y = (y | (y << 16n)) & 0x1f0000ff0000ffn;
    y = (y | (y << 8n)) & 0x100f00f00f00f00fn;
    y = (y | (y << 4n)) & 0x10c30c30c30c30c3n;
    y = (y | (y << 2n)) & 0x1249249249249249n;
    return y;
}

/** Mirrors SDFVoxelCommon::encodeLocation (level-local coords + level -> u64). */
export function encodeLocation(x: number, y: number, z: number, level: number): bigint {
    const lvl = BigInt(level);
    // levelLocalToGlobalCoords: (coord & mask) << (kMaxLevel - level).
    const g = (c: number) => (BigInt(c >>> 0) & ((1n << kMaxLevel) - 1n)) << (kMaxLevel - lvl);
    const sx = shiftCoord(Number(g(x)));
    const sy = shiftCoord(Number(g(y)));
    const sz = shiftCoord(Number(g(z)));
    return (lvl << kLocationCodeLevelOffset) | (((sx << 2n) | (sy << 1n) | sz) & kLocationCodeVoxelCoordsMask);
}

/** Mirrors SDFVoxelCommon::createChildLocationCode. */
export function createChildLocationCode(code: bigint, childID: number): bigint {
    const level = 1n + ((code >> kLocationCodeLevelOffset) & 0x1fn);
    const lvl = level < kMaxLevel ? level : kMaxLevel;
    let bits = code & kLocationCodeVoxelCoordsMask;
    bits |= BigInt(childID & 0x7) << (kLocationCodeLevelOffset - 3n * lvl);
    bits &= kLocationCodeVoxelCoordsMask;
    bits |= lvl << kLocationCodeLevelOffset;
    return bits;
}

/** Level bits from a location code. */
export function decodeLevel(code: bigint): number {
    return Number((code >> kLocationCodeLevelOffset) & 0x1fn);
}

const kMarker = 1 << 31; // locationCode.y hash-table marker bit set during the build.

interface SvoNode {
    code: bigint; // location code WITH the marker bit (as stored)
    validMask: number;
    pv0: number; // packedValues.x (x=0 corners)
    pv1: number; // packedValues.y (x=1 corners)
}

export class SDFSVO {
    gridWidth = 0;
    levelCount = 0;
    /** snorm8 dense field, (gridWidth+1)^3, x-fastest. */
    sdField = new Int8Array(0);
    /** SDFSVOVoxel records, 24-byte stride (relationData, pad, locX, locY, pv0, pv1). */
    svoData = new Uint32Array(0);
    voxelCount = 0;

    /** Mirrors SDFGrid::generateCheeseValues + SDFSVO::setValuesInternal. */
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.setValues(generateCheeseCornerValues(gridWidth, seed), gridWidth);
        this.build();
    }

    /** Mirrors SDFSVO::setValuesInternal (snorm8, normFactor gw/(0.5*sqrt3)). */
    setValues(cornerValues: Float32Array, gridWidth: number): void {
        this.gridWidth = gridWidth;
        this.levelCount = bitScanReverse(gridWidth) + 1;
        const gwv = gridWidth + 1;
        this.sdField = new Int8Array(gwv * gwv * gwv);
        const normalizationMultiplier = gridWidth / (0.5 * kSqrt3);
        for (let v = 0; v < this.sdField.length; v++) {
            const normalized = Math.min(Math.max(cornerValues[v]! * normalizationMultiplier, -1), 1);
            const integerScale = normalized * 127;
            this.sdField[v] = Math.trunc(integerScale >= 0 ? integerScale + 0.5 : integerScale - 0.5);
        }
    }

    /** snorm8 byte (uint8) of the field at a grid coord. */
    private byteAt(x: number, y: number, z: number): number {
        const gwv = this.gridWidth + 1;
        return this.sdField[x + gwv * (y + gwv * z)]! & 0xff;
    }

    /** int8 value (sign) of the field at a grid coord. */
    private valAt(x: number, y: number, z: number): number {
        const b = this.byteAt(x, y, z);
        return b < 128 ? b : b - 256;
    }

    /** True if the 8 corners at (x0,y0,z0) with the given step straddle the surface. */
    private containsSurface(x0: number, y0: number, z0: number, step: number): boolean {
        let anyNeg = false;
        let anyPos = false;
        for (let c = 0; c < 8; c++) {
            const v = this.valAt(x0 + (c & 1) * step, y0 + ((c >> 1) & 1) * step, z0 + ((c >> 2) & 1) * step);
            if (v <= 0) anyNeg = true;
            if (v >= 0) anyPos = true;
        }
        return anyNeg && anyPos;
    }

    /** packedValues (x0 corners, x1 corners) at (x0,y0,z0) with the given step. */
    private packCorners(x0: number, y0: number, z0: number, step: number): [number, number] {
        // packValues order: values0xx = x=0 corners (000,001,010,011); values1xx = x=1.
        const b = (dx: number, dy: number, dz: number) => this.byteAt(x0 + dx * step, y0 + dy * step, z0 + dz * step);
        const pv0 = (b(0, 0, 0) | (b(0, 0, 1) << 8) | (b(0, 1, 0) << 16) | (b(0, 1, 1) << 24)) >>> 0;
        const pv1 = (b(1, 0, 0) | (b(1, 0, 1) << 8) | (b(1, 1, 0) << 16) | (b(1, 1, 1) << 24)) >>> 0;
        return [pv0, pv1];
    }

    /** Bottom-up sparse octree build (mirrors the SDFSVO build kernels on CPU). */
    private build(): void {
        const gw = this.gridWidth;
        const numLevels = this.levelCount;
        const finest = numLevels - 1;
        const nodesByCode = new Map<string, SvoNode>();
        const nodes: SvoNode[] = [];

        for (let level = finest; level >= 0; level--) {
            const levelWidth = 1 << level;
            const hierarchy = numLevels - level - 1;
            const voxelWidth = 1 << hierarchy;
            for (let vz = 0; vz < levelWidth; vz++) {
                for (let vy = 0; vy < levelWidth; vy++) {
                    for (let vx = 0; vx < levelWidth; vx++) {
                        let validMask = 0;
                        if (level === finest && !this.containsSurface(vx, vy, vz, 1)) continue;
                        const code = encodeLocation(vx, vy, vz, level) | (BigInt(kMarker) << 32n);
                        if (level !== finest) {
                            // A voxel exists if any of its 8 children (level+1)
                            // exists. Address children through createChildLocationCode
                            // so the octant<->coord convention matches the build
                            // kernels exactly (octant bit2=x, bit1=y, bit0=z).
                            for (let c = 0; c < 8; c++) {
                                const childCode = (createChildLocationCode(code, c) | (BigInt(kMarker) << 32n)).toString();
                                if (nodesByCode.has(childCode)) validMask |= 1 << c;
                            }
                            if (validMask === 0) continue;
                        }
                        const gx = vx << hierarchy;
                        const gy = vy << hierarchy;
                        const gz = vz << hierarchy;
                        const [pv0, pv1] = this.packCorners(gx, gy, gz, voxelWidth);
                        const node: SvoNode = { code, validMask, pv0, pv1 };
                        nodes.push(node);
                        nodesByCode.set(code.toString(), node);
                    }
                }
            }
        }

        // Sort ascending by (locationCode.y high 32, then .x low 32) — the
        // bitonic sort order used by SDFSVOLocationCodeSorter. Root (level 0)
        // has the smallest .y, so it sorts to svoOffset 0 (the traversal root).
        const lo = (c: bigint) => Number(c & 0xffffffffn);
        const hi = (c: bigint) => Number((c >> 32n) & 0xffffffffn);
        nodes.sort((a, b) => hi(a.code) - hi(b.code) || lo(a.code) - lo(b.code));

        const offsetByCode = new Map<string, number>();
        nodes.forEach((n, i) => offsetByCode.set(n.code.toString(), i));

        this.voxelCount = nodes.length;
        this.svoData = new Uint32Array(nodes.length * 6); // 24-byte stride
        nodes.forEach((n, i) => {
            let relationData = n.validMask & 0xff;
            // firstValidChild's svoOffset in the upper 24 bits.
            for (let c = 0; c < 8; c++) {
                if (n.validMask & (1 << c)) {
                    const childCode = (createChildLocationCode(n.code, c) | (BigInt(kMarker) << 32n)).toString();
                    const childOffset = offsetByCode.get(childCode) ?? 0;
                    relationData |= childOffset << 8;
                    break;
                }
            }
            this.svoData[i * 6 + 0] = relationData >>> 0;
            // [i*6+1] = padding
            this.svoData[i * 6 + 2] = lo(n.code) >>> 0; // locationCode.x
            this.svoData[i * 6 + 3] = hi(n.code) >>> 0; // locationCode.y (with marker)
            this.svoData[i * 6 + 4] = n.pv0;
            this.svoData[i * 6 + 5] = n.pv1;
        });
    }
}
