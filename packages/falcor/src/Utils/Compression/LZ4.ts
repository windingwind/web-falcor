/**
 * LZ4 block-format decompression.
 *
 * Web substitute for a library native Falcor links against (OpenVDB pulls in
 * c-blosc, whose default sub-codec is LZ4): browsers ship deflate but no LZ4,
 * so the block format is decoded here. Only decompression is implemented —
 * nothing in the port writes LZ4.
 *
 * Block format: a sequence of [token][literal length bytes][literals]
 * [2-byte match offset][match length bytes]. The token's high nibble is the
 * literal length and its low nibble the match length minus 4; a nibble of 15
 * means "add the following 0xFF-terminated byte run". LZ4HC produces the same
 * block format.
 */

import { RuntimeError } from "../../Core/Error.js";

/**
 * Decompresses one LZ4 block.
 *
 * @param src Compressed bytes (the whole array is one block).
 * @param destSize Exact uncompressed size (the container always knows it).
 */
export function lz4Decompress(src: Uint8Array, destSize: number): Uint8Array {
    const dst = new Uint8Array(destSize);
    const end = src.length;
    let ip = 0;
    let op = 0;

    while (ip < end) {
        const token = src[ip++]!;

        // Literals.
        let literalLength = token >> 4;
        if (literalLength === 15) {
            let b = 255;
            while (b === 255) {
                if (ip >= end) throw new RuntimeError("LZ4: truncated literal length");
                b = src[ip++]!;
                literalLength += b;
            }
        }
        if (ip + literalLength > end || op + literalLength > destSize) throw new RuntimeError("LZ4: literal run overruns the buffer");
        for (let i = 0; i < literalLength; i++) dst[op++] = src[ip++]!;

        // The last sequence is literals-only and stops here.
        if (ip >= end) break;

        // Match.
        if (ip + 2 > end) throw new RuntimeError("LZ4: truncated match offset");
        const offset = src[ip]! | (src[ip + 1]! << 8);
        ip += 2;
        if (offset === 0 || offset > op) throw new RuntimeError(`LZ4: bad match offset ${offset} at output ${op}`);

        let matchLength = token & 0x0f;
        if (matchLength === 15) {
            let b = 255;
            while (b === 255) {
                if (ip >= end) throw new RuntimeError("LZ4: truncated match length");
                b = src[ip++]!;
                matchLength += b;
            }
        }
        matchLength += 4; // MINMATCH

        if (op + matchLength > destSize) throw new RuntimeError("LZ4: match overruns the buffer");
        // Byte-wise copy: matches may overlap the output cursor (run-length encoding).
        let match = op - offset;
        for (let i = 0; i < matchLength; i++) dst[op++] = dst[match++]!;
    }

    if (op !== destSize) throw new RuntimeError(`LZ4: decompressed ${op} bytes, expected ${destSize}`);
    return dst;
}
