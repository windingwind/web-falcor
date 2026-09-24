/**
 * Block-compression encoders for ImageIO.saveToDDS: the web stand-in for NVTT (native
 * compresses through NVIDIA Texture Tools). BC1–BC5 fit endpoints along the block's
 * principal axis and refine them by least squares; BC7 encodes mode 6 (RGBA, one subset,
 * 4-bit indices, p-bits searched); BC6H encodes mode 11 (one region, 10-bit endpoints)
 * for the signed and unsigned formats. §9: the output is valid BC, not NVTT's bit pattern.
 *
 * Inputs are RGBA float texels (w*h*4, block-linear rows); partial edge blocks replicate
 * the last row/column. Values are the stored ones: [0,1] for the UNORM formats, [-1,1] for
 * BC4/BC5 SNORM, and half-representable floats for BC6H.
 */

export type BCFormat = "BC1" | "BC2" | "BC3" | "BC4" | "BC4S" | "BC5" | "BC5S" | "BC6H" | "BC6HS" | "BC7";

const blockBytes: Record<BCFormat, number> = { BC1: 8, BC2: 16, BC3: 16, BC4: 8, BC4S: 8, BC5: 16, BC5S: 16, BC6H: 16, BC6HS: 16, BC7: 16 };

/** Compressed size of one level. */
export function bcLevelSize(format: BCFormat, width: number, height: number): number {
    return Math.ceil(width / 4) * Math.ceil(height / 4) * blockBytes[format];
}

/** Encodes one level. */
export function encodeBC(format: BCFormat, rgba: Float32Array, width: number, height: number): Uint8Array {
    const bx = Math.ceil(width / 4);
    const by = Math.ceil(height / 4);
    const size = blockBytes[format];
    const out = new Uint8Array(bx * by * size);
    const block = new Float32Array(64);
    for (let y = 0; y < by; y++) {
        for (let x = 0; x < bx; x++) {
            for (let t = 0; t < 16; t++) {
                const px = Math.min(x * 4 + (t & 3), width - 1);
                const py = Math.min(y * 4 + (t >> 2), height - 1);
                block.set(rgba.subarray((py * width + px) * 4, (py * width + px) * 4 + 4), t * 4);
            }
            const dst = out.subarray((y * bx + x) * size, (y * bx + x + 1) * size);
            encodeBlock(format, block, dst);
        }
    }
    return out;
}

function encodeBlock(format: BCFormat, b: Float32Array, dst: Uint8Array): void {
    switch (format) {
        case "BC1":
            encodeColor(b, dst, 0);
            return;
        case "BC2":
            for (let t = 0; t < 16; t += 2) {
                const lo = Math.round(clamp01(b[t * 4 + 3]!) * 15);
                const hi = Math.round(clamp01(b[(t + 1) * 4 + 3]!) * 15);
                dst[t >> 1] = lo | (hi << 4);
            }
            encodeColor(b, dst, 8);
            return;
        case "BC3":
            encodeChannel(b, 3, false, dst, 0);
            encodeColor(b, dst, 8);
            return;
        case "BC4":
        case "BC4S":
            encodeChannel(b, 0, format === "BC4S", dst, 0);
            return;
        case "BC5":
        case "BC5S":
            encodeChannel(b, 0, format === "BC5S", dst, 0);
            encodeChannel(b, 1, format === "BC5S", dst, 8);
            return;
        case "BC6H":
        case "BC6HS":
            encodeBC6HMode11(b, format === "BC6HS", dst);
            return;
        case "BC7":
            encodeBC7Mode6(b, dst);
            return;
    }
}

const clamp01 = (v: number) => (v > 0 ? (v < 1 ? v : 1) : 0);

/** Principal axis of `n`-channel points (power iteration on the covariance). */
function principalAxis(points: number[][], n: number): { mean: number[]; axis: number[] } {
    const mean = new Array<number>(n).fill(0);
    for (const p of points) for (let c = 0; c < n; c++) mean[c]! += p[c]! / points.length;
    const cov = Array.from({ length: n }, () => new Array<number>(n).fill(0));
    for (const p of points) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) cov[i]![j]! += (p[i]! - mean[i]!) * (p[j]! - mean[j]!);
    let axis = new Array<number>(n).fill(1);
    for (let it = 0; it < 8; it++) {
        const next = cov.map((row) => row.reduce((s, v, j) => s + v * axis[j]!, 0));
        const len = Math.hypot(...next);
        if (len < 1e-20) break;
        axis = next.map((v) => v / len);
    }
    return { mean, axis };
}

/**
 * Endpoints along the principal axis, refined by least squares for a palette of `levels`
 * evenly weighted entries; `quantize` snaps an endpoint, `palette` builds the decoded
 * entries. Returns the best endpoints and indices found.
 */
function fitEndpoints(
    points: number[][],
    n: number,
    weights: number[],
    quantize: (e: number[]) => number[],
    palette: (e0: number[], e1: number[]) => number[][],
    error: (a: number[], b: number[]) => number,
): { e0: number[]; e1: number[]; indices: number[]; err: number } {
    const { mean, axis } = principalAxis(points, n);
    let lo = Infinity;
    let hi = -Infinity;
    for (const p of points) {
        const d = p.reduce((s, v, c) => s + (v - mean[c]!) * axis[c]!, 0);
        lo = Math.min(lo, d);
        hi = Math.max(hi, d);
    }
    let e0 = quantize(mean.map((m, c) => m + axis[c]! * lo));
    let e1 = quantize(mean.map((m, c) => m + axis[c]! * hi));
    const assign = (a: number[], b: number[]) => {
        const pal = palette(a, b);
        let total = 0;
        const idx = points.map((p) => {
            let best = 0;
            let bestErr = Infinity;
            pal.forEach((q, i) => {
                const e = error(p, q);
                if (e < bestErr) {
                    bestErr = e;
                    best = i;
                }
            });
            total += bestErr;
            return best;
        });
        return { idx, total };
    };
    let { idx, total } = assign(e0, e1);
    let best = { e0, e1, indices: idx, err: total };
    for (let iter = 0; iter < 3; iter++) {
        // Least squares: p ≈ (1 - w) e0 + w e1 per texel.
        let aa = 0, ab = 0, bb = 0;
        const ax = new Array<number>(n).fill(0);
        const bx = new Array<number>(n).fill(0);
        points.forEach((p, t) => {
            const w = weights[idx[t]!]!;
            const a = 1 - w;
            aa += a * a;
            ab += a * w;
            bb += w * w;
            for (let c = 0; c < n; c++) {
                ax[c]! += a * p[c]!;
                bx[c]! += w * p[c]!;
            }
        });
        const det = aa * bb - ab * ab;
        if (Math.abs(det) < 1e-12) break;
        e0 = quantize(ax.map((v, c) => (bb * v - ab * bx[c]!) / det));
        e1 = quantize(bx.map((v, c) => (aa * v - ab * ax[c]!) / det));
        ({ idx, total } = assign(e0, e1));
        if (total < best.err) best = { e0, e1, indices: idx, err: total };
        else break;
    }
    return best;
}

const sq = (a: number[], b: number[]) => a.reduce((s, v, i) => s + (v - b[i]!) * (v - b[i]!), 0);

// --- BC1 color (also the color half of BC2/BC3) ---

const expand5 = (v: number) => (v << 3) | (v >> 2);
const expand6 = (v: number) => (v << 2) | (v >> 4);

function encodeColor(b: Float32Array, dst: Uint8Array, at: number): void {
    const points = Array.from({ length: 16 }, (_, t) => [clamp01(b[t * 4]!) * 255, clamp01(b[t * 4 + 1]!) * 255, clamp01(b[t * 4 + 2]!) * 255]);
    const to565 = (e: number[]) => [Math.round((clamp(e[0]!, 0, 255) * 31) / 255), Math.round((clamp(e[1]!, 0, 255) * 63) / 255), Math.round((clamp(e[2]!, 0, 255) * 31) / 255)];
    const decode = (q: number[]) => [expand5(q[0]!), expand6(q[1]!), expand5(q[2]!)];
    // Endpoints stay in 565 units; the palette decodes like the hardware (and DDSLoader).
    const fit = fitEndpoints(
        points,
        3,
        [0, 1, 1 / 3, 2 / 3],
        (e) => decode(to565(e)),
        (e0, e1) => [e0, e1, e0.map((v, c) => (2 * v + e1[c]!) / 3), e0.map((v, c) => (v + 2 * e1[c]!) / 3)],
        sq,
    );
    let c0 = pack565(to565(fit.e0));
    let c1 = pack565(to565(fit.e1));
    let indices = fit.indices;
    // Four-color mode needs c0 > c1: swap, remapping 0<->1 and 2<->3.
    if (c0 < c1) {
        [c0, c1] = [c1, c0];
        indices = indices.map((i) => i ^ 1);
    } else if (c0 === c1) indices = indices.map(() => 0);
    dst[at] = c0 & 0xff;
    dst[at + 1] = c0 >> 8;
    dst[at + 2] = c1 & 0xff;
    dst[at + 3] = c1 >> 8;
    let bits = 0;
    indices.forEach((i, t) => (bits |= i << (2 * t)));
    for (let k = 0; k < 4; k++) dst[at + 4 + k] = (bits >>> (8 * k)) & 0xff;
}

const pack565 = (q: number[]) => (q[0]! << 11) | (q[1]! << 5) | q[2]!;
const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v);

// --- BC4 channel (BC3 alpha, BC4, BC5) ---

function encodeChannel(b: Float32Array, channel: number, signed: boolean, dst: Uint8Array, at: number): void {
    const lo = signed ? -127 : 0;
    const hi = signed ? 127 : 255;
    const values = Array.from({ length: 16 }, (_, t) => clamp(b[t * 4 + channel]! * (signed ? 127 : 255), lo, hi));
    // Eight-value mode (a0 > a1): endpoints plus six interpolants.
    const fit = fitEndpoints(
        values.map((v) => [v]),
        1,
        [0, 1, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7],
        (e) => [clamp(Math.round(e[0]!), lo, hi)],
        (e0, e1) => [e0, e1, ...[1, 2, 3, 4, 5, 6].map((i) => [((7 - i) * e0[0]! + i * e1[0]!) / 7])],
        sq,
    );
    let a0 = fit.e0[0]!;
    let a1 = fit.e1[0]!;
    let indices = fit.indices;
    const remap8 = [1, 0, 7, 6, 5, 4, 3, 2];
    if (a0 < a1) {
        [a0, a1] = [a1, a0];
        indices = indices.map((i) => remap8[i]!);
    } else if (a0 === a1) indices = indices.map(() => 0);
    const byte = (v: number) => (signed ? v & 0xff : v);
    dst[at] = byte(a0);
    dst[at + 1] = byte(a1);
    let lo24 = 0;
    let hi24 = 0;
    indices.forEach((i, t) => {
        if (t < 8) lo24 |= i << (3 * t);
        else hi24 |= i << (3 * (t - 8));
    });
    for (let k = 0; k < 3; k++) {
        dst[at + 2 + k] = (lo24 >>> (8 * k)) & 0xff;
        dst[at + 5 + k] = (hi24 >>> (8 * k)) & 0xff;
    }
}

// --- BC7 mode 6 ---

const kWeights4 = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

class BitWriter {
    private bit = 0;
    constructor(private readonly out: Uint8Array) {
        out.fill(0);
    }
    write(value: number, count: number): void {
        for (let i = 0; i < count; i++, this.bit++) if ((value >>> i) & 1) this.out[this.bit >> 3]! |= 1 << (this.bit & 7);
    }
}

function encodeBC7Mode6(b: Float32Array, dst: Uint8Array): void {
    const points = Array.from({ length: 16 }, (_, t) => [0, 1, 2, 3].map((c) => clamp01(b[t * 4 + c]!) * 255));
    let best: { q0: number[]; q1: number[]; p0: number; p1: number; indices: number[]; err: number } | null = null;
    for (const p0 of [0, 1]) {
        for (const p1 of [0, 1]) {
            const quant = (p: number) => (e: number[]) => e.map((v) => ((clamp(Math.round((v - p) / 2), 0, 127) << 1) | p));
            // The palette interpolates the 8-bit endpoints ((7-bit << 1) | p).
            const palette = (e0: number[], e1: number[]) => kWeights4.map((w) => e0.map((v, c) => ((64 - w) * v + w * e1[c]! + 32) >> 6));
            // fitEndpoints quantizes both endpoints alike; refit each with its own p-bit.
            const fit = fitEndpoints(points, 4, kWeights4.map((w) => w / 64), quant(p0), palette, sq);
            const e0 = quant(p0)(fit.e0);
            const e1 = quant(p1)(fit.e1);
            const pal = palette(e0, e1);
            let err = 0;
            const indices = points.map((pt) => {
                let bi = 0;
                let be = Infinity;
                pal.forEach((q, i) => {
                    const e = sq(pt, q);
                    if (e < be) {
                        be = e;
                        bi = i;
                    }
                });
                err += be;
                return bi;
            });
            if (!best || err < best.err) best = { q0: e0.map((v) => v >> 1), q1: e1.map((v) => v >> 1), p0, p1, indices, err };
        }
    }
    let { q0, q1, p0, p1, indices } = best!;
    // The anchor (texel 0) index has an implicit 0 top bit: swap endpoints if needed.
    if (indices[0]! >= 8) {
        [q0, q1] = [q1, q0];
        [p0, p1] = [p1, p0];
        indices = indices.map((i) => 15 - i);
    }
    const w = new BitWriter(dst);
    w.write(1 << 6, 7); // mode 6
    for (let c = 0; c < 4; c++) {
        w.write(q0[c]!, 7);
        w.write(q1[c]!, 7);
    }
    w.write(p0, 1);
    w.write(p1, 1);
    indices.forEach((i, t) => w.write(i, t === 0 ? 3 : 4));
}

// --- BC6H mode 11 ---

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** float -> half bits (round to nearest even), clamped to the largest finite half. */
export function toHalf(v: number): number {
    if (Number.isNaN(v)) return 0x7bff;
    const sign = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
    const a = Math.abs(v);
    if (a >= 65504) return sign | 0x7bff;
    if (a < 2 ** -14) {
        // Subnormal: multiples of 2^-24 (Math.round-half-even).
        const q = a * 2 ** 24;
        const r = Math.round(q);
        return sign | (r - q === 0.5 && r % 2 === 1 ? r - 1 : r);
    }
    f32[0] = a;
    const x = u32[0]!;
    const e = ((x >>> 23) & 0xff) - 127 + 15;
    let h = (e << 10) | ((x >>> 13) & 0x3ff);
    const rem = x & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1) === 1)) h++;
    return sign | Math.min(h, 0x7bff);
}

/**
 * Mode 11: one region, 10-bit endpoints, 4-bit indices. The fit runs in the domain the
 * hardware interpolates in: half-bit magnitudes scaled by 64/31 (unsigned) or ±32/31
 * (signed), which the final `* 31 >> 6` / `* 31 >> 5` maps back to half bits.
 */
function encodeBC6HMode11(b: Float32Array, signed: boolean, dst: Uint8Array): void {
    const toDomain = (v: number) => {
        const h = toHalf(signed ? v : Math.max(0, v));
        const mag = (h & 0x7fff) * (signed ? 32 / 31 : 64 / 31);
        return h & 0x8000 ? -mag : mag;
    };
    const points = Array.from({ length: 16 }, (_, t) => [0, 1, 2].map((c) => toDomain(b[t * 4 + c]!)));
    const minComp = signed ? -511 : 0;
    const maxComp = signed ? 511 : 1023;
    const unquantize = (comp: number) => {
        if (!signed) {
            if (comp === 0) return 0;
            if (comp === 1023) return 0xffff;
            return ((comp << 16) + 0x8000) >> 10;
        }
        const c = Math.abs(comp);
        const u = c === 0 ? 0 : c >= 511 ? 0x7fff : ((c << 15) + 0x4000) >> 9;
        return comp < 0 ? -u : u;
    };
    const quantizeComp = (v: number) => {
        const guess = Math.round(signed ? (v * 512) / 32768 : (v * 1024) / 65536);
        let bestC = clamp(guess, minComp, maxComp);
        let bestE = Math.abs(unquantize(bestC) - v);
        for (let c = guess - 2; c <= guess + 2; c++) {
            const cc = clamp(c, minComp, maxComp);
            const e = Math.abs(unquantize(cc) - v);
            if (e < bestE) {
                bestE = e;
                bestC = cc;
            }
        }
        return bestC;
    };
    // Endpoints are carried as unquantized values; the palette is the hardware interpolation.
    const toUnq = (e: number[]) => e.map((v) => unquantize(quantizeComp(v)));
    const palette = (e0: number[], e1: number[]) => kWeights4.map((w) => e0.map((v, c) => Math.floor(((64 - w) * v + w * e1[c]! + 32) / 64)));
    const fit = fitEndpoints(points, 3, kWeights4.map((w) => w / 64), toUnq, palette, sq);
    let q0 = fit.e0.map(quantizeComp);
    let q1 = fit.e1.map(quantizeComp);
    let indices = fit.indices;
    if (indices[0]! >= 8) {
        [q0, q1] = [q1, q0];
        indices = indices.map((i) => 15 - i);
    }
    const w = new BitWriter(dst);
    w.write(0b00011, 5);
    const bits = (v: number) => v & 0x3ff; // two's complement for the signed format
    for (const q of [q0, q1]) for (let c = 0; c < 3; c++) w.write(bits(q[c]!), 10);
    indices.forEach((i, t) => w.write(i, t === 0 ? 3 : 4));
}
