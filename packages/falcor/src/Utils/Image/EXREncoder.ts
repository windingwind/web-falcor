/**
 * Minimal OpenEXR encoder: single-part scanline image, RGBA float32,
 * NO_COMPRESSION (the web capture path; native writes via OpenEXR).
 * Round-trips bit-exactly through decodeExr.
 */

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

/** Encodes top-down RGBA float32 pixels as an uncompressed scanline EXR. */
export function encodeExr(data: Float32Array, width: number, height: number): Uint8Array {
    const w = new Writer();
    w.i32(kMagic);
    w.i32(2); // version 2, single-part scanline

    // channels: alphabetical (A, B, G, R), each float (type 2), sampling 1.
    const channelEntry = 18; // name(2) + type(4) + pLinear+reserved(4) + xSampling(4) + ySampling(4)
    w.attr("channels", "chlist", 4 * channelEntry + 1);
    for (const name of ["A", "B", "G", "R"]) {
        w.str(name);
        w.i32(2); // FLOAT
        w.i32(0); // pLinear + reserved
        w.i32(1); // xSampling
        w.i32(1); // ySampling
    }
    w.u8(0); // end of channel list

    w.attr("compression", "compression", 1);
    w.u8(0); // NO_COMPRESSION
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

    // Scanline offset table (u64 per scanline), then blocks: y(i32),
    // size(i32), channel-planar pixel data (A,B,G,R rows).
    const scanBytes = width * 4 * 4;
    const blockSize = 8 + scanBytes;
    const tableStart = w.length;
    const dataStart = tableStart + height * 8;
    for (let y = 0; y < height; y++) w.u64(dataStart + y * blockSize);

    for (let y = 0; y < height; y++) {
        w.i32(y);
        w.i32(scanBytes);
        const row = new Float32Array(width * 4);
        for (const [c, srcC] of [[0, 3], [1, 2], [2, 1], [3, 0]] as const) {
            for (let x = 0; x < width; x++) row[c * width + x] = data[(y * width + x) * 4 + srcC]!;
        }
        w.bytes(new Uint8Array(row.buffer));
    }
    return w.concat();
}
