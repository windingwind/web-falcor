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

export class SDFSVO {
    gridWidth = 0;
    levelCount = 0;
    /** snorm8 dense field, (gridWidth+1)^3, x-fastest. */
    sdField = new Int8Array(0);

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
}
