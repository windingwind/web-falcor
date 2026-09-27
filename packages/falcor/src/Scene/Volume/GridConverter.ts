/**
 * Mirrors Scene/Volume/GridConverter.h (NanoVDBToBricksConverter): a NanoVDB float grid as bricks of 8^3 voxels,
 * with a 4-mip range texture (f16 majorant/minorant per brick), an indirection texture and a brick atlas.
 */

import { ResourceFormat } from "../../Core/API/Formats.js";
import { float16ToFloat32, float32ToFloat16 } from "../../Utils/Math/Float16.js";
import { compressAlphaDxt5 } from "./BC4Encode.js";
import { NanoVDBAccessor } from "./NanoVDBAccessor.js";

/** BrickedGrid's texture contents (native BrickedGrid holds the textures made from them). */
export interface BrickedGridData {
    /** Range and indirection texture size at mip 0 (bricks). */
    leafDim: [number, number, number];
    /** Range texture mips 0-3, RG16Float texels packed as (majorant | minorant << 16). */
    rangeMips: Uint32Array[];
    /** RGBA8Uint texels: atlas brick x + (y << 8) + (z << 16). */
    indirection: Uint32Array;
    /** Atlas size in texels. */
    atlasSize: [number, number, number];
    atlasFormat: ResourceFormat;
    /** BC4 blocks, or R8/R16 texels. */
    atlas: Uint8Array;
    nonEmptyCount: number;
}

const kBrickSize = 8;
const f = Math.fround;
const f32tof16 = (v: number) => float32ToFloat16(v);
const f16tof32 = (v: number) => float16ToFloat32(v & 0xffff);

/** NanoVDBToBricksConverter<TexelType, kBitsPerTexel>::convert for 4 (BC4, native's choice), 8 or 16 bits. */
export function convertNanoVDBToBricks(gridBuffer: Uint8Array, bitsPerTexel: 4 | 8 | 16 = 4): BrickedGridData {
    const a = new NanoVDBAccessor(gridBuffer);
    const bc4 = bitsPerTexel === 4;
    const { min, max } = a.indexBBox;
    const bbMin = min.map((v) => v & ~7);
    const bbMax = max.map((v) => (v + 7) & ~7);
    // Four mips: the coarsest covers 8 * 2^3 = 64 voxels.
    const pixDim = bbMax.map((v, k) => (v - bbMin[k]! + 63) & ~63);
    const leafDim = [0, 1, 2, 3].map((i) => pixDim.map((v) => v / (8 << i)) as [number, number, number]);
    const mipCount = leafDim.map(([x, y, z]) => x * y * z);
    const leafCount = a.leafCount;
    // The first two atlas dimensions are powers of 2 (f32 math, as natively).
    const approxdim = 1 << Math.trunc(f(f(Math.log2(f(leafCount + 1))) / 3));
    const lastdim = Math.ceil(leafCount / (approxdim * approxdim));
    const atlasBricks: [number, number, number] = [approxdim, approxdim, lastdim];
    const atlasPx: [number, number, number] = [approxdim * kBrickSize, approxdim * kBrickSize, lastdim * kBrickSize];
    const brickMax = approxdim * approxdim * lastdim;
    const texelCount = atlasPx[0] * atlasPx[1] * atlasPx[2];
    const bytesPerTexel = bitsPerTexel === 16 ? 2 : 1;
    const atlas = new Uint8Array(bc4 ? texelCount / 2 : texelCount * bytesPerTexel);
    const atlas16 = bitsPerTexel === 16 ? new Uint16Array(atlas.buffer) : null;
    const range0 = new Uint32Array(mipCount[0]!);
    const indirection = new Uint32Array(mipCount[0]!);
    const pixelsPerSlice = atlasPx[0] * atlasPx[1];
    const bricksPerSlice = atlasBricks[0] * atlasBricks[1];
    let nonEmpty = 0;

    // Majorant/minorant over the brick and its 1-voxel halo (native's getValue loops cover the whole 10^3 block):
    // each of the 27 neighbouring 8^3 blocks is a leaf or a constant tile value.
    const expandBlock = (ox: number, oy: number, oz: number, mm: [number, number]) => {
        for (let dx = -1; dx <= 1; dx++)
            for (let dy = -1; dy <= 1; dy++)
                for (let dz = -1; dz <= 1; dz++) {
                    const [bx, by, bz] = [ox + dx * 8, oy + dy * 8, oz + dz * 8];
                    const leaf = a.probeLeaf(bx, by, bz);
                    // The voxel range of this block inside [-1, 8]^3 (local to the block).
                    const [x0, x1] = dx < 0 ? [7, 7] : dx > 0 ? [0, 0] : [0, 7];
                    const [y0, y1] = dy < 0 ? [7, 7] : dy > 0 ? [0, 0] : [0, 7];
                    const [z0, z1] = dz < 0 ? [7, 7] : dz > 0 ? [0, 0] : [0, 7];
                    if (leaf < 0) {
                        const v = a.tileValue;
                        if (v < mm[0]) mm[0] = v;
                        if (v > mm[1]) mm[1] = v;
                        continue;
                    }
                    const vals = a.leafValues(leaf);
                    for (let x = x0; x <= x1; x++)
                        for (let y = y0; y <= y1; y++)
                            for (let z = z0; z <= z1; z++) {
                                const v = vals[(x << 6) | (y << 3) | z]!;
                                if (v < mm[0]) mm[0] = v;
                                if (v > mm[1]) mm[1] = v;
                            }
                }
    };

    const tile = new Uint8Array(16);
    let o = 0;
    for (let z = 0; z < leafDim[0]![2]; z++) {
        for (let y = 0; y < leafDim[0]![1]; y++) {
            for (let x = 0; x < leafDim[0]![0]; x++, o++) {
                const [ix, iy, iz] = [x * 8 + bbMin[0]!, y * 8 + bbMin[1]!, z * 8 + bbMin[2]!];
                const val = a.getValue(ix, iy, iz);
                const leaf = a.probeLeaf(ix, iy, iz);
                const mm: [number, number] = [val, val];
                let myleaf = 0;
                if (leaf >= 0) {
                    expandBlock(ix, iy, iz, mm);
                    if (mm[0] !== mm[1]) myleaf = nonEmpty++;
                }
                let [minorant, majorant] = mm;
                if (majorant === minorant || myleaf >= brickMax || leaf < 0) {
                    // Identical majorant and minorant.
                    range0[o] = (f32tof16(majorant) + f32tof16(majorant) * 65536) >>> 0;
                    indirection[o] = 0;
                    continue;
                }
                const data = a.leafValues(leaf);
                majorant = f16tof32(f32tof16(majorant) + 1);
                minorant = f16tof32(f32tof16(minorant));
                range0[o] = (f32tof16(majorant) + f32tof16(minorant) * 65536) >>> 0;
                const atlasx = myleaf % atlasBricks[0];
                const atlasy = Math.floor(myleaf / atlasBricks[0]) % atlasBricks[1];
                const atlasz = Math.floor(myleaf / bricksPerSlice);
                indirection[o] = (atlasx + (atlasy << 8) + (atlasz << 16)) >>> 0;
                // uint8_t/uint16_t(float): truncation, wrapping out-of-range values as the x86 conversion does.
                const mask = bitsPerTexel === 16 ? 0xffff : 0xff;
                if (!bc4) {
                    const invRange = f(((1 << bitsPerTexel) - 1) / f(majorant - minorant));
                    let dst = atlasx * kBrickSize + atlasy * (atlasPx[0] * kBrickSize) + atlasz * (pixelsPerSlice * kBrickSize);
                    for (let pz = 0; pz < kBrickSize; pz++, dst += pixelsPerSlice - atlasPx[0] * kBrickSize)
                        for (let py = 0; py < kBrickSize; py++, dst += atlasPx[0] - kBrickSize)
                            for (let px = 0; px < kBrickSize; px++, dst++) {
                                const t = Math.trunc(f(f(data[px * 64 + py * 8 + pz]! - minorant) * invRange)) & mask;
                                if (atlas16) atlas16[dst] = t;
                                else atlas[dst] = t;
                            }
                    continue;
                }
                const invRange = f(255 / f(majorant - minorant));
                // In 8-byte blocks: 4x4 texel tiles, row-major per z slice.
                let dst = atlasx * (kBrickSize / 4) + atlasy * ((atlasPx[0] / 4) * (kBrickSize / 4)) + atlasz * ((pixelsPerSlice / 16) * kBrickSize);
                for (let pz = 0; pz < kBrickSize; pz++, dst += pixelsPerSlice / 16 - (atlasPx[0] / 4) * (kBrickSize / 4)) {
                    for (let ty = 0; ty < kBrickSize; ty += 4, dst += atlasPx[0] / 4 - kBrickSize / 4) {
                        for (let tx = 0; tx < kBrickSize; tx += 4, dst++) {
                            for (let py = 0; py < 4; py++)
                                for (let px = 0; px < 4; px++) tile[py * 4 + px] = Math.trunc(f(f(data[(px + tx) * 64 + (py + ty) * 8 + pz]! - minorant) * invRange)) & 0xff;
                            compressAlphaDxt5(tile, atlas, dst * 8);
                        }
                    }
                }
            }
        }
    }

    // Mips 1-3: each texel combines 2x2x2 of the finer mip (max of majorants, min of minorants).
    const rangeMips = [range0];
    for (let mip = 1; mip < 4; mip++) {
        const src = rangeMips[mip - 1]!;
        const [sx, sy] = leafDim[mip - 1]!;
        const [tx, ty, tz] = leafDim[mip]!;
        const dst = new Uint32Array(mipCount[mip]!);
        const row = sx;
        const slice = sx * sy;
        let s = 0;
        let d = 0;
        for (let z = 0; z < tz; z++, s += slice) {
            for (let y = 0; y < ty; y++, s += row) {
                for (let x = 0; x < tx; x++, s += 2) {
                    let maj = -Infinity;
                    let mn = Infinity;
                    for (const at of [s, s + 1, s + row, s + 1 + row, s + slice, s + slice + 1, s + slice + row, s + slice + 1 + row]) {
                        maj = Math.max(maj, f16tof32(src[at]!));
                        mn = Math.min(mn, f16tof32(src[at]! >>> 16));
                    }
                    dst[d++] = (f32tof16(maj) + f32tof16(mn) * 65536) >>> 0;
                }
            }
        }
        rangeMips.push(dst);
    }

    return {
        leafDim: leafDim[0]!,
        rangeMips,
        indirection,
        atlasSize: atlasPx,
        atlasFormat: bc4 ? ResourceFormat.BC4Unorm : bitsPerTexel === 8 ? ResourceFormat.R8Unorm : ResourceFormat.R16Unorm,
        atlas,
        nonEmptyCount: nonEmpty,
    };
}

/** A BC4 3D atlas (rows of 4x4 blocks per z slice) decoded to floats, for adapters without BC 3D textures. */
export function decodeBC4Volume(blocks: Uint8Array, width: number, height: number, depth: number): Float32Array {
    const out = new Float32Array(width * height * depth);
    const [bw, bh] = [width / 4, height / 4];
    const palette = new Float32Array(8);
    for (let z = 0; z < depth; z++)
        for (let by = 0; by < bh; by++)
            for (let bx = 0; bx < bw; bx++) {
                const o = ((z * bh + by) * bw + bx) * 8;
                const [r0, r1] = [blocks[o]!, blocks[o + 1]!];
                palette[0] = r0 / 255;
                palette[1] = r1 / 255;
                // BC4 UNORM: 6 interpolated values, or 4 plus 0 and 1.
                if (r0 > r1) for (let i = 2; i < 8; i++) palette[i] = ((8 - i) * r0 + (i - 1) * r1) / (7 * 255);
                else {
                    for (let i = 2; i < 6; i++) palette[i] = ((6 - i) * r0 + (i - 1) * r1) / (5 * 255);
                    [palette[6], palette[7]] = [0, 1];
                }
                // 16 3-bit indices, little-endian over bytes 2-7.
                let bits = 0n;
                for (let k = 7; k >= 2; k--) bits = (bits << 8n) | BigInt(blocks[o + k]!);
                for (let t = 0; t < 16; t++) {
                    const x = bx * 4 + (t & 3);
                    const y = by * 4 + (t >> 2);
                    out[(z * height + y) * width + x] = palette[Number((bits >> BigInt(3 * t)) & 7n)]!;
                }
            }
    return out;
}
