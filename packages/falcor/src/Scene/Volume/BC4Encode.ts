/**
 * Mirrors Scene/Volume/BC4Encode.h (derived from libsquish's alpha.cpp, MIT, (c) 2006 Simon Brown):
 * CompressAlphaDxt5 encodes a 4x4 tile of uint8 values into one 8-byte BC4 block.
 */

function fixRange(min: number, max: number, steps: number): [number, number] {
    if (max - min < steps) max = Math.min(min + steps, 255);
    if (max - min < steps) min = Math.max(0, max - steps);
    return [min, max];
}

/** Fits each value to the codebook; returns the summed squared error. */
function fitCodes(tile: Uint8Array, codes: Uint8Array, indices: Uint8Array): number {
    let err = 0;
    for (let i = 0; i < 16; i++) {
        const value = tile[i]!;
        let least = Number.MAX_SAFE_INTEGER;
        let index = 0;
        for (let j = 0; j < 8; j++) {
            const dist = (value - codes[j]!) ** 2;
            if (dist < least) {
                least = dist;
                index = j;
            }
        }
        indices[i] = index;
        err += least;
    }
    return err;
}

function writeAlphaBlock(alpha0: number, alpha1: number, indices: Uint8Array, out: Uint8Array, o: number): void {
    out[o] = alpha0;
    out[o + 1] = alpha1;
    let dest = o + 2;
    for (let i = 0; i < 2; i++) {
        // 8 3-bit indices in 3 bytes.
        let value = 0;
        for (let j = 0; j < 8; j++) value |= indices[i * 8 + j]! << (3 * j);
        for (let j = 0; j < 3; j++) out[dest++] = (value >> (8 * j)) & 0xff;
    }
}

function writeAlphaBlock5(alpha0: number, alpha1: number, indices: Uint8Array, out: Uint8Array, o: number): void {
    if (alpha0 > alpha1) {
        const swapped = indices.map((index) => (index === 0 ? 1 : index === 1 ? 0 : index <= 5 ? 7 - index : index));
        writeAlphaBlock(alpha1, alpha0, swapped, out, o);
    } else writeAlphaBlock(alpha0, alpha1, indices, out, o);
}

function writeAlphaBlock7(alpha0: number, alpha1: number, indices: Uint8Array, out: Uint8Array, o: number): void {
    if (alpha0 < alpha1) {
        const swapped = indices.map((index) => (index === 0 ? 1 : index === 1 ? 0 : 9 - index));
        writeAlphaBlock(alpha1, alpha0, swapped, out, o);
    } else writeAlphaBlock(alpha0, alpha1, indices, out, o);
}

/** Encodes `tile` (16 values, row-major) as a BC4 block at out[o..o+8). */
export function compressAlphaDxt5(tile: Uint8Array, out: Uint8Array, o: number): void {
    let [min5, max5, min7, max7] = [255, 0, 255, 0];
    for (let i = 0; i < 16; i++) {
        const value = tile[i]!;
        if (value < min7) min7 = value;
        if (value > max7) max7 = value;
        if (value !== 0 && value < min5) min5 = value;
        if (value !== 255 && value > max5) max5 = value;
    }
    if (min5 > max5) min5 = max5;
    if (min7 > max7) min7 = max7;
    [min5, max5] = fixRange(min5, max5, 5);
    [min7, max7] = fixRange(min7, max7, 7);

    const codes5 = new Uint8Array(8);
    codes5[0] = min5;
    codes5[1] = max5;
    for (let i = 1; i < 5; i++) codes5[1 + i] = Math.trunc(((5 - i) * min5 + i * max5) / 5);
    codes5[6] = 0;
    codes5[7] = 255;
    const codes7 = new Uint8Array(8);
    codes7[0] = min7;
    codes7[1] = max7;
    for (let i = 1; i < 7; i++) codes7[1 + i] = Math.trunc(((7 - i) * min7 + i * max7) / 7);

    const indices5 = new Uint8Array(16);
    const indices7 = new Uint8Array(16);
    const err5 = fitCodes(tile, codes5, indices5);
    const err7 = fitCodes(tile, codes7, indices7);
    if (err5 <= err7) writeAlphaBlock5(min5, max5, indices5, out, o);
    else writeAlphaBlock7(min7, max7, indices7, out, o);
}
