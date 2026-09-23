/**
 * PNG decode/encode — the web stand-in for FreeImage's PNG plugin inside
 * Bitmap. Browser decoders collapse everything to 8-bit RGBA; this keeps the
 * bit depth and colour type Bitmap needs to pick native's ResourceFormat.
 * zlib runs on the platform's (De)CompressionStream.
 */

import { RuntimeError } from "../../Core/Error.js";

export interface PngImage {
    width: number;
    height: number;
    /** 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA (palettes expand to 3 or 4). */
    channels: number;
    /** 8 or 16 (lower depths expand to 8). */
    bitDepth: number;
    /** Whether the source was palettized (FreeImage converts those to 32 bits). */
    palettized: boolean;
    /** Top-down rows, tightly packed; 16-bit samples as native-endian Uint16. */
    data: Uint8Array | Uint16Array;
}

const kSignature = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

export function isPng(bytes: Uint8Array): boolean {
    return bytes.length >= 8 && kSignature.every((b, i) => bytes[i] === b);
}

async function pipe(data: Uint8Array, stream: CompressionStream | DecompressionStream): Promise<Uint8Array> {
    const out = new Response(new Blob([data as BlobPart]).stream().pipeThrough(stream));
    return new Uint8Array(await out.arrayBuffer());
}

export const zlibInflate = (data: Uint8Array) => pipe(data, new DecompressionStream("deflate"));
export const zlibDeflate = (data: Uint8Array) => pipe(data, new CompressionStream("deflate"));

/** zlib stream of stored (uncompressed) deflate blocks, as PNG_Z_NO_COMPRESSION writes. */
export function zlibStore(data: Uint8Array): Uint8Array {
    const blocks = Math.max(1, Math.ceil(data.length / 65535));
    const out = new Uint8Array(2 + data.length + blocks * 5 + 4);
    out[0] = 0x78;
    out[1] = 0x01;
    let o = 2;
    for (let b = 0; b < blocks; b++) {
        const chunk = data.subarray(b * 65535, Math.min(data.length, (b + 1) * 65535));
        out[o++] = b === blocks - 1 ? 1 : 0;
        out[o++] = chunk.length & 0xff;
        out[o++] = chunk.length >> 8;
        out[o++] = ~chunk.length & 0xff;
        out[o++] = (~chunk.length >> 8) & 0xff;
        out.set(chunk, o);
        o += chunk.length;
    }
    new DataView(out.buffer).setUint32(o, adler32(data));
    return out;
}

function adler32(data: Uint8Array): number {
    let a = 1;
    let b = 0;
    for (let i = 0; i < data.length; i++) {
        a = (a + data[i]!) % 65521;
        b = (b + a) % 65521;
    }
    return ((b << 16) | a) >>> 0;
}

const kCrcTable = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        t[n] = c >>> 0;
    }
    return t;
})();

function crc32(bytes: Uint8Array): number {
    let c = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) c = kCrcTable[(c ^ bytes[i]!) & 0xff]! ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
}

function paeth(a: number, b: number, c: number): number {
    const p = a + b - c;
    const pa = Math.abs(p - a);
    const pb = Math.abs(p - b);
    const pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

/** Undoes the per-scanline filters of one (sub)image in place; returns the raw rows. */
function unfilter(src: Uint8Array, offset: number, height: number, bpp: number, rowBytes: number): { rows: Uint8Array; end: number } {
    const rows = new Uint8Array(rowBytes * height);
    let o = offset;
    for (let y = 0; y < height; y++) {
        const type = src[o++]!;
        const row = rows.subarray(y * rowBytes, (y + 1) * rowBytes);
        const prev = y > 0 ? rows.subarray((y - 1) * rowBytes, y * rowBytes) : null;
        for (let x = 0; x < rowBytes; x++) {
            const raw = src[o++]!;
            const a = x >= bpp ? row[x - bpp]! : 0;
            const b = prev ? prev[x]! : 0;
            const c = prev && x >= bpp ? prev[x - bpp]! : 0;
            let v: number;
            if (type === 0) v = raw;
            else if (type === 1) v = raw + a;
            else if (type === 2) v = raw + b;
            else if (type === 3) v = raw + ((a + b) >> 1);
            else if (type === 4) v = raw + paeth(a, b, c);
            else throw new RuntimeError(`PNG: bad filter type ${type}`);
            row[x] = v & 0xff;
        }
    }
    return { rows, end: o };
}

/** FreeImage's gray-ramp test over a full 2^bpp palette (missing entries are black). */
export function isGrayRamp(rgb: Uint8Array, count: number): boolean {
    for (let i = 0; i < count; i++) {
        const v = Math.round((i * 255) / (count - 1));
        const [r, g, b] = i * 3 + 2 < rgb.length ? [rgb[i * 3]!, rgb[i * 3 + 1]!, rgb[i * 3 + 2]!] : [0, 0, 0];
        if (r !== v || g !== v || b !== v) return false;
    }
    return true;
}

export async function decodePng(bytes: Uint8Array): Promise<PngImage> {
    if (!isPng(bytes)) throw new RuntimeError("PNG: bad signature");
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let pos = 8;
    let width = 0;
    let height = 0;
    let depth = 0;
    let colorType = 0;
    let interlace = 0;
    let palette: Uint8Array | null = null;
    let trns: Uint8Array | null = null;
    const idat: Uint8Array[] = [];
    while (pos + 8 <= bytes.length) {
        const len = view.getUint32(pos);
        const type = String.fromCharCode(...bytes.subarray(pos + 4, pos + 8));
        const body = bytes.subarray(pos + 8, pos + 8 + len);
        pos += 12 + len;
        if (type === "IHDR") {
            width = view.getUint32(body.byteOffset - bytes.byteOffset);
            height = view.getUint32(body.byteOffset - bytes.byteOffset + 4);
            [depth, colorType] = [body[8]!, body[9]!];
            interlace = body[12]!;
        } else if (type === "PLTE") palette = body;
        else if (type === "tRNS") trns = body;
        else if (type === "IDAT") idat.push(body);
        else if (type === "IEND") break;
    }
    if (!(width > 0 && height > 0)) throw new RuntimeError("PNG: missing IHDR");
    const samples = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 }[colorType as 0 | 2 | 3 | 4 | 6];
    if (!samples) throw new RuntimeError(`PNG: bad colour type ${colorType}`);

    const compressed = new Uint8Array(idat.reduce((n, c) => n + c.length, 0));
    let o = 0;
    for (const c of idat) {
        compressed.set(c, o);
        o += c.length;
    }
    const inflated = await zlibInflate(compressed);

    const bitsPerPixel = samples * depth;
    const bpp = Math.max(1, bitsPerPixel >> 3);
    // Unpack one sample at (row, x, s) from a filtered-out row buffer.
    const sampleAt = (row: Uint8Array, x: number, s: number): number => {
        if (depth === 16) return (row[(x * samples + s) * 2]! << 8) | row[(x * samples + s) * 2 + 1]!;
        if (depth === 8) return row[x * samples + s]!;
        const bit = (x * samples + s) * depth;
        return (row[bit >> 3]! >> (8 - depth - (bit & 7))) & ((1 << depth) - 1);
    };

    // FreeImage honours tRNS on palettes only; truecolour/gray keys are ignored.
    // A palette that is a gray ramp reads as gray (FreeImage_GetColorType: FIC_MINISBLACK).
    const grayRamp = colorType === 3 && isGrayRamp(palette!, 1 << depth);
    const palettized = colorType === 3 && !grayRamp;
    const channels = palettized ? (trns ? 4 : 3) : samples;
    const outDepth = depth === 16 ? 16 : 8;
    const data = outDepth === 16 ? new Uint16Array(width * height * channels) : new Uint8Array(width * height * channels);
    const scale = !palettized && depth < 8 ? 255 / ((1 << depth) - 1) : 1;

    const store = (rows: Uint8Array, rowBytes: number, w: number, h: number, mapX: (x: number) => number, mapY: (y: number) => number) => {
        for (let y = 0; y < h; y++) {
            const row = rows.subarray(y * rowBytes, (y + 1) * rowBytes);
            for (let x = 0; x < w; x++) {
                const dst = (mapY(y) * width + mapX(x)) * channels;
                if (grayRamp) {
                    data[dst] = palette![sampleAt(row, x, 0) * 3]!;
                    continue;
                }
                if (palettized) {
                    const idx = sampleAt(row, x, 0);
                    data[dst] = palette![idx * 3]!;
                    data[dst + 1] = palette![idx * 3 + 1]!;
                    data[dst + 2] = palette![idx * 3 + 2]!;
                    if (channels === 4) data[dst + 3] = idx < trns!.length ? trns![idx]! : 255;
                    continue;
                }
                for (let s = 0; s < samples; s++) data[dst + s] = Math.round(sampleAt(row, x, s) * scale);
            }
        }
    };

    if (interlace === 0) {
        const rowBytes = Math.ceil((width * bitsPerPixel) / 8);
        const { rows } = unfilter(inflated, 0, height, bpp, rowBytes);
        store(rows, rowBytes, width, height, (x) => x, (y) => y);
    } else {
        // Adam7: seven passes over sub-lattices.
        const passes = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]] as const;
        let offset = 0;
        for (const [x0, y0, dx, dy] of passes) {
            const w = Math.ceil((width - x0) / dx);
            const h = Math.ceil((height - y0) / dy);
            if (w <= 0 || h <= 0) continue;
            const rowBytes = Math.ceil((w * bitsPerPixel) / 8);
            const { rows, end } = unfilter(inflated, offset, h, bpp, rowBytes);
            offset = end;
            store(rows, rowBytes, w, h, (x) => x0 + x * dx, (y) => y0 + y * dy);
        }
    }
    return { width, height, channels, bitDepth: outDepth, palettized, data };
}

/** Encodes 8-bit gray/RGB/RGBA top-down pixels (filter type 0 on every row). */
export async function encodePng(pixels: Uint8Array, width: number, height: number, channels: 1 | 3 | 4, compress: boolean): Promise<Uint8Array> {
    const rowBytes = width * channels;
    const raw = new Uint8Array((rowBytes + 1) * height);
    for (let y = 0; y < height; y++) raw.set(pixels.subarray(y * rowBytes, (y + 1) * rowBytes), y * (rowBytes + 1) + 1);
    const idat = compress ? await zlibDeflate(raw) : zlibStore(raw);

    const chunks: Uint8Array[] = [new Uint8Array(kSignature)];
    const chunk = (type: string, body: Uint8Array) => {
        const out = new Uint8Array(12 + body.length);
        const v = new DataView(out.buffer);
        v.setUint32(0, body.length);
        for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
        out.set(body, 8);
        v.setUint32(8 + body.length, crc32(out.subarray(4, 8 + body.length)));
        chunks.push(out);
    };
    const ihdr = new Uint8Array(13);
    const hv = new DataView(ihdr.buffer);
    hv.setUint32(0, width);
    hv.setUint32(4, height);
    ihdr[8] = 8;
    ihdr[9] = channels === 1 ? 0 : channels === 3 ? 2 : 6;
    chunk("IHDR", ihdr);
    chunk("IDAT", idat);
    chunk("IEND", new Uint8Array(0));
    const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
    let o = 0;
    for (const c of chunks) {
        out.set(c, o);
        o += c.length;
    }
    return out;
}
