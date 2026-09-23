/**
 * Minimal OpenEXR encoder: single-part scanline image, RGB or RGBA, HALF or
 * FLOAT, NO_COMPRESSION or ZIP_COMPRESSION (Bitmap.saveImage and the web
 * capture path; native writes via OpenEXR). Round-trips through decodeExr.
 */

import { float32ToFloat16 } from "../Math/Float16.js";
import { zlibDeflate } from "./PNGCodec.js";

const kMagic = 0x01312f76;

class Writer {
    private chunks: Uint8Array[] = [];
    private textEncoder = new TextEncoder();

    bytes(b: Uint8Array): void {
        this.chunks.push(b);
    }

    u8(v: number): void {
        this.bytes(new Uint8Array([v & 0xff]));
    }

    i32(v: number): void {
        const b = new Uint8Array(4);
        new DataView(b.buffer).setInt32(0, v, true);
        this.bytes(b);
    }

    f32(v: number): void {
        const b = new Uint8Array(4);
        new DataView(b.buffer).setFloat32(0, v, true);
        this.bytes(b);
    }

    u64(v: number): void {
        const b = new Uint8Array(8);
        new DataView(b.buffer).setBigUint64(0, BigInt(v), true);
        this.bytes(b);
    }

    /** NUL-terminated string. */
    str(s: string): void {
        this.bytes(this.textEncoder.encode(s));
        this.u8(0);
    }

    /** Header attribute: name, type, size, value(bytes appended by caller). */
    attr(name: string, type: string, size: number): void {
        this.str(name);
        this.str(type);
        this.i32(size);
    }

    concat(): Uint8Array {
        const total = this.chunks.reduce((acc, c) => acc + c.byteLength, 0);
        const out = new Uint8Array(total);
        let off = 0;
        for (const c of this.chunks) {
            out.set(c, off);
            off += c.byteLength;
        }
        return out;
    }

    get length(): number {
        return this.chunks.reduce((acc, c) => acc + c.byteLength, 0);
    }
}

export interface ExrEncodeOptions {
    /** Write the A channel (default true). */
    alpha?: boolean;
    /** HALF instead of FLOAT samples (default false). */
    half?: boolean;
}

/** Channel-planar scanline bytes for rows [y0, y1): each row holds (A,)B,G,R runs. */
function scanlineBytes(data: Float32Array, width: number, y0: number, y1: number, opts: Required<ExrEncodeOptions>): Uint8Array {
    const channels: [number, number][] = opts.alpha ? [[0, 3], [1, 2], [2, 1], [3, 0]] : [[0, 2], [1, 1], [2, 0]];
    const sampleBytes = opts.half ? 2 : 4;
    const rowBytes = width * channels.length * sampleBytes;
    const out = new Uint8Array(rowBytes * (y1 - y0));
    const view = new DataView(out.buffer);
    for (let y = y0; y < y1; y++) {
        for (const [c, srcC] of channels) {
            let o = (y - y0) * rowBytes + c * width * sampleBytes;
            for (let x = 0; x < width; x++, o += sampleBytes) {
                const v = data[(y * width + x) * 4 + srcC]!;
                if (opts.half) view.setUint16(o, float32ToFloat16(v), true);
                else view.setFloat32(o, v, true);
            }
        }
    }
    return out;
}

/** Header + offset table + blocks; `blocks[i]` is the (possibly compressed) data of block i. */
function assemble(width: number, height: number, opts: Required<ExrEncodeOptions>, compression: number, linesPerBlock: number, blocks: Uint8Array[]): Uint8Array {
    const w = new Writer();
    w.i32(kMagic);
    w.i32(2); // version 2, single-part scanline

    // channels: alphabetical ((A,) B, G, R), HALF (1) or FLOAT (2), sampling 1.
    const names = opts.alpha ? ["A", "B", "G", "R"] : ["B", "G", "R"];
    const channelEntry = 18; // name(2) + type(4) + pLinear+reserved(4) + xSampling(4) + ySampling(4)
    w.attr("channels", "chlist", names.length * channelEntry + 1);
    for (const name of names) {
        w.str(name);
        w.i32(opts.half ? 1 : 2);
        w.i32(0); // pLinear + reserved
        w.i32(1); // xSampling
        w.i32(1); // ySampling
    }
    w.u8(0); // end of channel list

    w.attr("compression", "compression", 1);
    w.u8(compression);
    w.attr("dataWindow", "box2i", 16);
    w.i32(0);
    w.i32(0);
    w.i32(width - 1);
    w.i32(height - 1);
    w.attr("displayWindow", "box2i", 16);
    w.i32(0);
    w.i32(0);
    w.i32(width - 1);
    w.i32(height - 1);
    w.attr("lineOrder", "lineOrder", 1);
    w.u8(0); // INCREASING_Y
    w.attr("pixelAspectRatio", "float", 4);
    w.f32(1);
    w.attr("screenWindowCenter", "v2f", 8);
    w.f32(0);
    w.f32(0);
    w.attr("screenWindowWidth", "float", 4);
    w.f32(1);
    w.u8(0); // end of header

    // Offset table (u64 per block), then blocks: y(i32), size(i32), data.
    let offset = w.length + blocks.length * 8;
    for (const b of blocks) {
        w.u64(offset);
        offset += 8 + b.length;
    }
    blocks.forEach((b, i) => {
        w.i32(i * linesPerBlock);
        w.i32(b.length);
        w.bytes(b);
    });
    return w.concat();
}

/** Encodes top-down RGBA float32 pixels as an uncompressed scanline EXR. */
export function encodeExr(data: Float32Array, width: number, height: number, options: ExrEncodeOptions = {}): Uint8Array {
    const opts = { alpha: options.alpha ?? true, half: options.half ?? false };
    const blocks = Array.from({ length: height }, (_, y) => scanlineBytes(data, width, y, y + 1, opts));
    return assemble(width, height, opts, 0, 1, blocks);
}

/**
 * ZIP_COMPRESSION (16-line blocks): OpenEXR's byte interleave and delta
 * predictor, then zlib. A block that doesn't shrink is stored raw, as OpenEXR does.
 */
export async function encodeExrZip(data: Float32Array, width: number, height: number, options: ExrEncodeOptions = {}): Promise<Uint8Array> {
    const opts = { alpha: options.alpha ?? true, half: options.half ?? false };
    const kLines = 16;
    const blocks: Uint8Array[] = [];
    for (let y0 = 0; y0 < height; y0 += kLines) {
        const raw = scanlineBytes(data, width, y0, Math.min(height, y0 + kLines), opts);
        const t = new Uint8Array(raw.length);
        const half = (raw.length + 1) >> 1;
        for (let i = 0; i < raw.length; i++) t[(i & 1) === 0 ? i >> 1 : half + (i >> 1)] = raw[i]!;
        let p = t[0]!;
        for (let i = 1; i < t.length; i++) {
            const d = (t[i]! - p + 128 + 256) & 0xff;
            p = t[i]!;
            t[i] = d;
        }
        const z = await zlibDeflate(t);
        blocks.push(z.length < raw.length ? z : raw);
    }
    return assemble(width, height, opts, 3, kLines, blocks);
}
