/**
 * DEFLATE (RFC 1951) and zlib (RFC 1950) decompression.
 *
 * Web substitute for zlib, which native Falcor links against (OpenVDB's ZIP
 * codec, and blosc's zlib sub-codec). The browser's `DecompressionStream` can
 * do this too, but only asynchronously; the loaders that need it are
 * synchronous parsers, so the algorithm is implemented directly here.
 *
 * Decoding follows the canonical "puff" structure: per-length symbol counts
 * plus a symbol table, walked one bit at a time. Only decompression exists —
 * nothing in the port writes DEFLATE streams.
 */

import { RuntimeError } from "../../Core/Error.js";

/** Length codes 257..285: base length and extra bits (RFC 1951 §3.2.5). */
const kLengthBase = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const kLengthExtra = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
/** Distance codes 0..29: base distance and extra bits. */
const kDistBase = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const kDistExtra = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
/** Order in which the code-length code lengths are stored in a dynamic block. */
const kCodeLengthOrder = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

interface Huffman {
    /** counts[len] = number of codes of that bit length. */
    counts: Int32Array;
    /** Symbols ordered by (length, symbol). */
    symbols: Int32Array;
}

class BitReader {
    private bitBuffer = 0;
    private bitCount = 0;
    pos = 0;

    constructor(private readonly data: Uint8Array) {}

    /** Reads `n` bits LSB-first. */
    bits(n: number): number {
        while (this.bitCount < n) {
            if (this.pos >= this.data.length) throw new RuntimeError("Inflate: out of input");
            this.bitBuffer |= this.data[this.pos++]! << this.bitCount;
            this.bitCount += 8;
        }
        const value = this.bitBuffer & ((1 << n) - 1);
        this.bitBuffer >>>= n;
        this.bitCount -= n;
        return value;
    }

    /** Drops the partial byte (stored blocks are byte-aligned). */
    alignToByte(): void {
        this.bitBuffer = 0;
        this.bitCount = 0;
    }
}

function buildHuffman(lengths: Uint8Array | number[], count: number): Huffman {
    const counts = new Int32Array(16);
    for (let i = 0; i < count; i++) counts[lengths[i]!]!++;
    counts[0] = 0; // unused symbols don't take part

    const offsets = new Int32Array(16);
    for (let len = 1; len < 16; len++) offsets[len] = offsets[len - 1]! + counts[len - 1]!;


    const symbols = new Int32Array(count);
    for (let sym = 0; sym < count; sym++) {
        const len = lengths[sym]!;
        if (len !== 0) symbols[offsets[len]!++] = sym;
    }
    return { counts, symbols };
}

function decodeSymbol(br: BitReader, h: Huffman): number {
    let code = 0;
    let first = 0;
    let index = 0;
    for (let len = 1; len < 16; len++) {
        code |= br.bits(1);
        const count = h.counts[len]!;
        if (code - first < count) return h.symbols[index + (code - first)]!;
        index += count;
        first = (first + count) << 1;
        code <<= 1;
    }
    throw new RuntimeError("Inflate: invalid Huffman code");
}

let fixedLiteral: Huffman | null = null;
let fixedDistance: Huffman | null = null;

function fixedTables(): { literal: Huffman; distance: Huffman } {
    if (!fixedLiteral || !fixedDistance) {
        const lit = new Uint8Array(288);
        lit.fill(8, 0, 144);
        lit.fill(9, 144, 256);
        lit.fill(7, 256, 280);
        lit.fill(8, 280, 288);
        fixedLiteral = buildHuffman(lit, 288);
        fixedDistance = buildHuffman(new Uint8Array(30).fill(5), 30);
    }
    return { literal: fixedLiteral, distance: fixedDistance };
}

/** Growable output that becomes an exactly-sized Uint8Array at the end. */
class Output {
    private buf: Uint8Array;
    length = 0;

    constructor(capacity: number) {
        this.buf = new Uint8Array(Math.max(capacity, 64));
    }

    private reserve(extra: number): void {
        if (this.length + extra <= this.buf.length) return;
        let cap = this.buf.length * 2;
        while (cap < this.length + extra) cap *= 2;
        const next = new Uint8Array(cap);
        next.set(this.buf.subarray(0, this.length));
        this.buf = next;
    }

    push(byte: number): void {
        this.reserve(1);
        this.buf[this.length++] = byte;
    }

    copyFrom(distance: number, length: number): void {
        if (distance > this.length) throw new RuntimeError(`Inflate: distance ${distance} exceeds output ${this.length}`);
        this.reserve(length);
        let from = this.length - distance;
        for (let i = 0; i < length; i++) this.buf[this.length++] = this.buf[from++]!;
    }

    appendRaw(data: Uint8Array): void {
        this.reserve(data.length);
        this.buf.set(data, this.length);
        this.length += data.length;
    }

    finish(): Uint8Array {
        return this.buf.subarray(0, this.length);
    }
}

/**
 * Inflates a raw DEFLATE stream (no zlib/gzip wrapper).
 *
 * @param src Compressed bytes.
 * @param expectedSize Uncompressed size when the container knows it (avoids regrowing).
 */
export function inflateRaw(src: Uint8Array, expectedSize = 0): Uint8Array {
    const br = new BitReader(src);
    const out = new Output(expectedSize || src.length * 4);

    for (;;) {
        const isFinal = br.bits(1);
        const type = br.bits(2);

        if (type === 0) {
            // Stored: byte-aligned LEN/NLEN then raw bytes.
            br.alignToByte();
            if (br.pos + 4 > src.length) throw new RuntimeError("Inflate: truncated stored block header");
            const len = src[br.pos]! | (src[br.pos + 1]! << 8);
            const nlen = src[br.pos + 2]! | (src[br.pos + 3]! << 8);
            br.pos += 4;
            if ((len ^ 0xffff) !== nlen) throw new RuntimeError("Inflate: stored block length mismatch");
            if (br.pos + len > src.length) throw new RuntimeError("Inflate: truncated stored block");
            out.appendRaw(src.subarray(br.pos, br.pos + len));
            br.pos += len;
        } else if (type === 1 || type === 2) {
            let literal: Huffman;
            let distance: Huffman;
            if (type === 1) {
                ({ literal, distance } = fixedTables());
            } else {
                const hlit = br.bits(5) + 257;
                const hdist = br.bits(5) + 1;
                const hclen = br.bits(4) + 4;
                const clLengths = new Uint8Array(19);
                for (let i = 0; i < hclen; i++) clLengths[kCodeLengthOrder[i]!] = br.bits(3);
                const clTable = buildHuffman(clLengths, 19);

                const lengths = new Uint8Array(hlit + hdist);
                for (let i = 0; i < hlit + hdist; ) {
                    const sym = decodeSymbol(br, clTable);
                    if (sym < 16) {
                        lengths[i++] = sym;
                    } else if (sym === 16) {
                        if (i === 0) throw new RuntimeError("Inflate: repeat with no previous length");
                        const prev = lengths[i - 1]!;
                        for (let r = br.bits(2) + 3; r > 0; r--) lengths[i++] = prev;
                    } else if (sym === 17) {
                        for (let r = br.bits(3) + 3; r > 0; r--) lengths[i++] = 0;
                    } else {
                        for (let r = br.bits(7) + 11; r > 0; r--) lengths[i++] = 0;
                    }
                }
                literal = buildHuffman(lengths.subarray(0, hlit), hlit);
                distance = buildHuffman(lengths.subarray(hlit), hdist);
            }

            for (;;) {
                const sym = decodeSymbol(br, literal);
                if (sym < 256) {
                    out.push(sym);
                } else if (sym === 256) {
                    break; // end of block
                } else {
                    const li = sym - 257;
                    if (li >= kLengthBase.length) throw new RuntimeError(`Inflate: invalid length code ${sym}`);
                    const length = kLengthBase[li]! + br.bits(kLengthExtra[li]!);
                    const di = decodeSymbol(br, distance);
                    if (di >= kDistBase.length) throw new RuntimeError(`Inflate: invalid distance code ${di}`);
                    out.copyFrom(kDistBase[di]! + br.bits(kDistExtra[di]!), length);
                }
            }
        } else {
            throw new RuntimeError("Inflate: reserved block type");
        }

        if (isFinal) break;
    }
    return out.finish();
}

/** Inflates a zlib stream (RFC 1950 header, Adler-32 trailer ignored). */
export function inflate(src: Uint8Array, expectedSize = 0): Uint8Array {
    if (src.length < 2) throw new RuntimeError("Inflate: not a zlib stream");
    const cmf = src[0]!;
    const flg = src[1]!;
    if ((cmf & 0x0f) !== 8) throw new RuntimeError(`Inflate: unsupported zlib compression method ${cmf & 0x0f}`);
    if (((cmf << 8) | flg) % 31 !== 0) throw new RuntimeError("Inflate: bad zlib header check");
    if (flg & 0x20) throw new RuntimeError("Inflate: preset dictionaries are not supported");
    return inflateRaw(src.subarray(2), expectedSize);
}
