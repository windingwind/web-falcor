/**
 * Blosc1 container decompression (c-blosc format version 2).
 *
 * Web substitute for c-blosc, which native Falcor pulls in through OpenVDB:
 * `.vdb` files written by OpenVDB 5+ store their leaf buffers as blosc chunks
 * (LZ4 sub-codec with byte shuffle by default), so reading real-world volumes
 * means decoding this container.
 *
 * Layout (blosc.h / blosc.c):
 *   byte 0    version (2)
 *   byte 1    version of the sub-codec's own format
 *   byte 2    flags: 0x01 byte shuffle, 0x02 memcpy'd, 0x04 bit shuffle,
 *             0x10 "don't split", bits 5-7 sub-codec id
 *   byte 3    typesize (shuffle granularity)
 *   4..7      nbytes (uncompressed size)
 *   8..11     blocksize
 *   12..15    cbytes (total compressed size)
 *   16..      per-block start offsets (int32 each), then the blocks
 *
 * Each block holds `nsplits` independently compressed streams, each prefixed by
 * an int32 compressed size; a stream whose compressed size equals its
 * uncompressed size is stored verbatim (`blosc_d`).
 */

import { RuntimeError } from "../../Core/Error.js";
import { lz4Decompress } from "./LZ4.js";
import { inflate } from "./Inflate.js";

const kHeaderSize = 16; // BLOSC_MAX_OVERHEAD / BLOSC_MIN_HEADER_LENGTH
const kFlagShuffle = 0x01;
const kFlagMemcpyed = 0x02;
const kFlagBitShuffle = 0x04;
const kFlagDontSplit = 0x10;
/** blosc.c: MAX_SPLITS / MIN_BUFFERSIZE — they decide whether a block is split. */
const kMaxSplits = 16;
const kMinBufferSize = 128;

const kCodecNames: Record<number, string> = { 0: "blosclz", 1: "lz4", 2: "lz4hc", 3: "snappy", 4: "zlib", 5: "zstd" };

export interface BloscHeader {
    version: number;
    flags: number;
    typesize: number;
    /** Uncompressed size in bytes. */
    nbytes: number;
    blocksize: number;
    /** Total compressed size, header included. */
    cbytes: number;
    codec: number;
    codecName: string;
}

/** Reads the 16-byte header (does not decompress). */
export function bloscReadHeader(src: Uint8Array): BloscHeader {
    if (src.length < kHeaderSize) throw new RuntimeError("Blosc: buffer shorter than the header");
    const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
    const flags = src[2]!;
    const codec = (flags >> 5) & 0x7;
    return {
        version: src[0]!,
        flags,
        typesize: src[3]!,
        nbytes: view.getUint32(4, true),
        blocksize: view.getUint32(8, true),
        cbytes: view.getUint32(12, true),
        codec,
        codecName: kCodecNames[codec] ?? String(codec),
    };
}

/** Mirrors blosc's generic byte unshuffle (the trailing `blocksize % typesize` bytes stay put). */
function unshuffle(typesize: number, blocksize: number, src: Uint8Array, dst: Uint8Array, dstOffset: number): void {
    const elements = Math.floor(blocksize / typesize);
    const remainder = blocksize % typesize;
    for (let j = 0; j < typesize; j++) {
        const base = j * elements;
        for (let i = 0; i < elements; i++) dst[dstOffset + i * typesize + j] = src[base + i]!;
    }
    for (let i = blocksize - remainder; i < blocksize; i++) dst[dstOffset + i] = src[i]!;
}

/** Mirrors blosc's generic bit unshuffle (bit-plane transpose within the block). */
function bitUnshuffle(typesize: number, blocksize: number, src: Uint8Array, dst: Uint8Array, dstOffset: number): void {
    const elemsPerBlock = Math.floor(blocksize / typesize);
    const bits = typesize * 8;
    const vectorized = elemsPerBlock * typesize;
    dst.fill(0, dstOffset, dstOffset + blocksize);
    for (let bit = 0; bit < bits; bit++) {
        const planeBase = bit * Math.floor(elemsPerBlock / 8);
        for (let i = 0; i < elemsPerBlock; i++) {
            const byte = src[planeBase + (i >> 3)]!;
            if ((byte >> (i & 7)) & 1) {
                const target = dstOffset + i * typesize + (bit >> 3);
                dst[target] = dst[target]! | (1 << (bit & 7));
            }
        }
    }
    for (let i = vectorized; i < blocksize; i++) dst[dstOffset + i] = src[i]!;
}

function decodeStream(codec: number, src: Uint8Array, destSize: number): Uint8Array {
    switch (codec) {
        case 1: // lz4
        case 2: // lz4hc — same block format
            return lz4Decompress(src, destSize);
        case 4: // zlib
            return inflate(src, destSize);
        default:
            throw new RuntimeError(`Blosc: unsupported sub-codec '${kCodecNames[codec] ?? codec}' (only lz4/lz4hc/zlib are implemented)`);
    }
}

/**
 * Decompresses a blosc1 chunk.
 *
 * @param src The chunk, starting at its 16-byte header.
 * @returns Exactly `header.nbytes` bytes.
 */
export function bloscDecompress(src: Uint8Array): Uint8Array {
    const header = bloscReadHeader(src);
    if (header.version !== 1 && header.version !== 2) throw new RuntimeError(`Blosc: unsupported format version ${header.version}`);
    if (src.length < header.cbytes) throw new RuntimeError(`Blosc: chunk truncated (${src.length} < ${header.cbytes})`);

    const dst = new Uint8Array(header.nbytes);
    if (header.flags & kFlagMemcpyed) {
        dst.set(src.subarray(kHeaderSize, kHeaderSize + header.nbytes));
        return dst;
    }

    const { blocksize, typesize, nbytes } = header;
    if (blocksize <= 0) throw new RuntimeError("Blosc: zero block size");
    const leftover = nbytes % blocksize;
    const nblocks = Math.floor(nbytes / blocksize) + (leftover ? 1 : 0);

    const view = new DataView(src.buffer, src.byteOffset, src.byteLength);
    const dontSplit = (header.flags & kFlagDontSplit) !== 0;
    const doShuffle = (header.flags & kFlagShuffle) !== 0 && typesize > 1;
    const doBitShuffle = (header.flags & kFlagBitShuffle) !== 0 && blocksize >= typesize;
    // Shuffled blocks land in scratch first, then get transposed into place.
    const scratch = doShuffle || doBitShuffle ? new Uint8Array(blocksize) : null;

    for (let b = 0; b < nblocks; b++) {
        const isLeftover = b === nblocks - 1 && leftover > 0;
        const bsize = isLeftover ? leftover : blocksize;
        const nsplits = !dontSplit && typesize <= kMaxSplits && Math.floor(bsize / typesize) >= kMinBufferSize && !isLeftover ? typesize : 1;
        const neblock = Math.floor(bsize / nsplits);

        let offset = view.getInt32(kHeaderSize + b * 4, true);
        let written = 0;
        for (let s = 0; s < nsplits; s++) {
            if (offset < 0 || offset + 4 > src.length) throw new RuntimeError("Blosc: block offset out of range");
            const cbytes = view.getInt32(offset, true);
            offset += 4;
            if (cbytes < 0 || offset + cbytes > src.length) throw new RuntimeError("Blosc: split size out of range");
            const chunk = src.subarray(offset, offset + cbytes);
            const target = scratch ?? dst;
            const targetOffset = scratch ? written : b * blocksize + written;
            if (cbytes === neblock) {
                target.set(chunk, targetOffset); // stored verbatim (incompressible)
            } else {
                target.set(decodeStream(header.codec, chunk, neblock), targetOffset);
            }
            offset += cbytes;
            written += neblock;
        }

        if (scratch) {
            if (doShuffle) unshuffle(typesize, bsize, scratch, dst, b * blocksize);
            else bitUnshuffle(typesize, bsize, scratch, dst, b * blocksize);
        }
    }
    return dst;
}
