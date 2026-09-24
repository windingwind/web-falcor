/**
 * Port of Scene/SDFs/SparseBrickSet/BC4Encode.slang (compressBlock: 4x4 snorm8 values against a
 * 5- and a 7-interpolant codebook, keeping the one with less squared error) plus the BC4 SNORM
 * decode. A compressed SBS is uploaded as a real BC4Snorm texture (the GPU decodes it as it does
 * for native; its interpolation isn't exactly 1/7 steps), except when it shares the packed brick
 * texture with uncompressed grids, where its bricks take the CPU round trip below.
 */

function fixRange(lo: number, hi: number, steps: number): [number, number] {
    if (hi - lo < steps) {
        hi = Math.min(lo + steps, 127);
        lo = hi - lo < steps ? Math.max(-128, hi - steps) : lo;
    }
    return [lo, hi];
}

function fitCodes(block: Int32Array, codes: number[]): { err: number; indices: number[] } {
    let err = 0;
    const indices: number[] = [];
    for (let i = 0; i < 16; i++) {
        let least = 0x7fffffff;
        let index = 0;
        for (let j = 0; j < 8; j++) {
            const d = block[i]! - codes[j]!;
            if (d * d < least) {
                least = d * d;
                index = j;
            }
        }
        indices.push(index);
        err += least;
    }
    return { err, indices };
}

/** compressBlock: returns the block's endpoints and 16 indices as written (after the endpoint swap). */
export function compressBC4Block(block: Int32Array): { alpha0: number; alpha1: number; indices: number[] } {
    let [min5, max5, min7, max7] = [127, -128, 127, -128];
    for (const v of block) {
        min7 = Math.min(min7, v);
        max7 = Math.max(max7, v);
        if (v !== -128 && v < min5) min5 = v;
        if (v !== 127 && v > max5) max5 = v;
    }
    min5 = Math.min(min5, max5);
    min7 = Math.min(min7, max7);
    [min5, max5] = fixRange(min5, max5, 5);
    [min7, max7] = fixRange(min7, max7, 7);
    const div = (a: number, b: number) => Math.trunc(a / b); // C integer division
    const codes5 = [min5, max5, ...[1, 2, 3, 4].map((i) => div((5 - i) * min5 + i * max5, 5)), -128, 127];
    const codes7 = [min7, max7, ...[1, 2, 3, 4, 5, 6].map((i) => div((7 - i) * min7 + i * max7, 7))];
    const f5 = fitCodes(block, codes5);
    const f7 = fitCodes(block, codes7);
    if (f5.err <= f7.err) {
        // writeAlphaBlock5: alpha0 <= alpha1 selects the 6-interpolant decode.
        if (min5 > max5) return { alpha0: max5, alpha1: min5, indices: f5.indices.map((x) => (x === 0 ? 1 : x === 1 ? 0 : x <= 5 ? 7 - x : x)) };
        return { alpha0: min5, alpha1: max5, indices: f5.indices };
    }
    // writeAlphaBlock7: alpha0 > alpha1 selects the 8-interpolant decode.
    if (min7 < max7) return { alpha0: max7, alpha1: min7, indices: f7.indices.map((x) => (x === 0 ? 1 : x === 1 ? 0 : 9 - x)) };
    return { alpha0: min7, alpha1: max7, indices: f7.indices };
}

/** BC4 SNORM decode of one block (D3D rules: -128 reads as -1; 6 interpolants plus -1/1 when alpha0 <= alpha1). */
export function decodeBC4Block(alpha0: number, alpha1: number, indices: number[]): Float32Array {
    const e0 = Math.max(alpha0 / 127, -1);
    const e1 = Math.max(alpha1 / 127, -1);
    const palette =
        alpha0 > alpha1
            ? [e0, e1, ...[1, 2, 3, 4, 5, 6].map((i) => ((7 - i) * e0 + i * e1) / 7)]
            : [e0, e1, ...[1, 2, 3, 4].map((i) => ((5 - i) * e0 + i * e1) / 5), -1, 1];
    return Float32Array.from(indices, (i) => palette[i]!);
}

/** snorm8 values of one aligned 4x4 texel block, quantized like SDFSBSCreateBricksFromSDField (value * 127, rounded half away from 0). */
function quantizeBlock(texture: Float32Array, width: number, bx: number, by: number, block: Int32Array): void {
    for (let y = 0; y < 4; y++)
        for (let x = 0; x < 4; x++) {
            const s = Math.fround(texture[bx + x + width * (by + y)]! * 127);
            block[y * 4 + x] = Math.trunc(s >= 0 ? Math.fround(s + 0.5) : Math.fround(s - 0.5));
        }
}

/** A BC4Snorm texture's blocks (row-major, 8 bytes each) for a float texture whose width and height are multiples of 4. */
export function encodeBC4Texture(texture: Float32Array, width: number, height: number): Uint8Array {
    const out = new Uint8Array((width / 4) * (height / 4) * 8);
    const block = new Int32Array(16);
    let o = 0;
    for (let by = 0; by < height; by += 4)
        for (let bx = 0; bx < width; bx += 4, o += 8) {
            quantizeBlock(texture, width, bx, by, block);
            const { alpha0, alpha1, indices } = compressBC4Block(block);
            out.set(packBC4Block(alpha0, alpha1, indices), o);
        }
    return out;
}

/**
 * Replaces each 4x4 texel block of an R32Float brick texture (width, height multiples of 4) by its
 * BC4 round trip, quantizing like SDFSBSCreateBricksFromSDField (value * 127, rounded half away from 0).
 */
export function bc4RoundTripTexture(texture: Float32Array, width: number, height: number): void {
    const block = new Int32Array(16);
    for (let by = 0; by < height; by += 4)
        for (let bx = 0; bx < width; bx += 4) {
            quantizeBlock(texture, width, bx, by, block);
            const { alpha0, alpha1, indices } = compressBC4Block(block);
            const decoded = decodeBC4Block(alpha0, alpha1, indices);
            for (let y = 0; y < 4; y++) for (let x = 0; x < 4; x++) texture[bx + x + width * (by + y)] = decoded[y * 4 + x]!;
        }
}

/** writeAlphaBlock's 8-byte layout: the endpoints as snorm8, then 16 3-bit indices from bit 16. */
export function packBC4Block(alpha0: number, alpha1: number, indices: number[]): Uint8Array {
    const bytes = new Uint8Array(8);
    bytes[0] = alpha0 & 0xff;
    bytes[1] = alpha1 & 0xff;
    for (let i = 0; i < 16; i++) {
        const bit = 16 + 3 * (i % 8) + 24 * Math.floor(i / 8);
        for (let b = 0; b < 3; b++) if ((indices[i]! >> b) & 1) bytes[(bit + b) >> 3]! |= 1 << ((bit + b) & 7);
    }
    return bytes;
}
