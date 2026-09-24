/**
 * Block-compression encoders for ImageIO.saveToDDS: the web stand-in for NVTT (native
 * compresses through NVIDIA Texture Tools). BC1–BC5 fit endpoints along the block's
 * principal axis and refine them by least squares; BC7 searches all eight modes (ranked
 * partitions, p-bits per endpoint, alpha rotations) and keeps the least-error one; BC6H encodes mode 11 (one region, 10-bit endpoints)
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
            encodeBC7(b, dst);
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

// --- BC7 (modes 0-3, 6, 7) ---

const kWeights2 = [0, 21, 43, 64];
const kWeights3 = [0, 9, 18, 27, 37, 46, 55, 64];
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

// BC7 partition tables (D3D11 spec; via DirectXTex g_aPartitionTable / g_aFixUp): one digit per texel.
const kPartitions2 = ["0011001100110011", "0001000100010001", "0111011101110111", "0001001100110111", "0000000100010011", "0011011101111111", "0001001101111111", "0000000100110111", "0000000000010011", "0011011111111111", "0000000101111111", "0000000000010111", "0001011111111111", "0000000011111111", "0000111111111111", "0000000000001111", "0000100011101111", "0111000100000000", "0000000010001110", "0111001100010000", "0011000100000000", "0000100011001110", "0000000010001100", "0111001100110001", "0011000100010000", "0000100010001100", "0110011001100110", "0011011001101100", "0001011111101000", "0000111111110000", "0111000110001110", "0011100110011100", "0101010101010101", "0000111100001111", "0101101001011010", "0011001111001100", "0011110000111100", "0101010110101010", "0110100101101001", "0101101010100101", "0111001111001110", "0001001111001000", "0011001001001100", "0011101111011100", "0110100110010110", "0011110011000011", "0110011010011001", "0000011001100000", "0100111001000000", "0010011100100000", "0000001001110010", "0000010011100100", "0110110010010011", "0011011011001001", "0110001110011100", "0011100111000110", "0110110011001001", "0110001100111001", "0111111010000001", "0001100011100111", "0000111100110011", "0011001111110000", "0010001011101110", "0100010001110111"];
const kPartitions3 = ["0011001102212222", "0001001122112221", "0000200122112211", "0222002200110111", "0000000011221122", "0011001100220022", "0022002211111111", "0011001122112211", "0000000011112222", "0000111111112222", "0000111122222222", "0012001200120012", "0112011201120112", "0122012201220122", "0011011211221222", "0011200122002220", "0001001101121122", "0111001120012200", "0000112211221122", "0022002200221111", "0111011102220222", "0001000122212221", "0000001101220122", "0000110022102210", "0122012200110000", "0012001211222222", "0110122112210110", "0000011012211221", "0022110211020022", "0110011020022222", "0011012201220011", "0000200022112221", "0000000211221222", "0222002200120011", "0011001200220222", "0120012001200120", "0000111122220000", "0120120120120120", "0120201212010120", "0011220011220011", "0011112222000011", "0101010122222222", "0000000021212121", "0022112200221122", "0022001100220011", "0220122102201221", "0101222222220101", "0000212121212121", "0101010101012222", "0222011102220111", "0002111200021112", "0000211221122112", "0222011101110222", "0002111211120002", "0110011001102222", "0000000021122112", "0110011022222222", "0022001100110022", "0022112211220022", "0000000000002112", "0002000100020001", "0222122202221222", "0101222222222222", "0111201122012220"];
/** Anchor texel of subset 1 in the two-subset partitions. */
const kAnchor2 = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15];
/** Anchor texels of subsets 1 and 2 in the three-subset partitions. */
const kAnchor3 = [[3, 15], [3, 8], [15, 8], [15, 3], [8, 15], [3, 15], [15, 3], [15, 8], [8, 15], [8, 15], [6, 15], [6, 15], [6, 15], [5, 15], [3, 15], [3, 8], [3, 15], [3, 8], [8, 15], [15, 3], [3, 15], [3, 8], [6, 15], [10, 8], [5, 3], [8, 15], [8, 6], [6, 10], [8, 15], [5, 15], [15, 10], [15, 8], [8, 15], [15, 3], [3, 15], [5, 10], [6, 10], [10, 8], [8, 9], [15, 10], [15, 6], [3, 15], [15, 8], [5, 15], [15, 3], [15, 6], [15, 6], [15, 8], [3, 15], [15, 3], [5, 15], [5, 15], [5, 15], [8, 15], [5, 15], [10, 15], [5, 15], [10, 15], [8, 15], [13, 15], [15, 3], [12, 15], [3, 15], [3, 8]];

/** One BC7 mode's layout: subsets, partition bits, endpoint bits, p-bits and index bits. */
interface BC7Mode {
    mode: number;
    subsets: 1 | 2 | 3;
    partitionBits: number;
    colorBits: number;
    /** 0: RGB only (alpha decodes to 255). */
    alphaBits: number;
    pbits: "none" | "shared" | "unique";
    indexBits: number;
}

const kBC7Modes: BC7Mode[] = [
    { mode: 0, subsets: 3, partitionBits: 4, colorBits: 4, alphaBits: 0, pbits: "unique", indexBits: 3 },
    { mode: 1, subsets: 2, partitionBits: 6, colorBits: 6, alphaBits: 0, pbits: "shared", indexBits: 3 },
    { mode: 2, subsets: 3, partitionBits: 6, colorBits: 5, alphaBits: 0, pbits: "none", indexBits: 2 },
    { mode: 3, subsets: 2, partitionBits: 6, colorBits: 7, alphaBits: 0, pbits: "unique", indexBits: 2 },
    { mode: 6, subsets: 1, partitionBits: 0, colorBits: 7, alphaBits: 7, pbits: "unique", indexBits: 4 },
    { mode: 7, subsets: 2, partitionBits: 6, colorBits: 5, alphaBits: 5, pbits: "unique", indexBits: 2 },
];

/** Decoded 8-bit value of an n-bit endpoint component (with its p-bit appended, if any). */
function bc7Decode(q: number, bits: number, p: number): number {
    const n = p >= 0 ? bits + 1 : bits;
    const c = p >= 0 ? (q << 1) | p : q;
    const v = c << (8 - n);
    return v | (v >> n);
}

/** Nearest n-bit component for an 8-bit target (p-bit fixed). */
function bc7Quantize(v: number, bits: number, p: number): number {
    const max = (1 << bits) - 1;
    const guess = Math.round((v / 255) * max);
    let best = 0;
    let bestErr = Infinity;
    for (let q = Math.max(0, guess - 1); q <= Math.min(max, guess + 1); q++) {
        const e = Math.abs(bc7Decode(q, bits, p) - v);
        if (e < bestErr) {
            bestErr = e;
            best = q;
        }
    }
    return best;
}

interface BC7Subset {
    q: [number[], number[]];
    p: [number, number];
    indices: number[];
    err: number;
}

/** Fits one subset's endpoints, p-bits chosen per endpoint (shared: per subset). */
function bc7FitSubset(points: number[][], m: BC7Mode, n: number): BC7Subset {
    const weights = m.indexBits === 2 ? kWeights2 : m.indexBits === 3 ? kWeights3 : kWeights4;
    const bitsOf = (c: number) => (c < 3 ? m.colorBits : m.alphaBits);
    const pCombos: [number, number][] = m.pbits === "none" ? [[-1, -1]] : m.pbits === "shared" ? [[0, 0], [1, 1]] : [[0, 0], [0, 1], [1, 0], [1, 1]];
    const palette = (e0: number[], e1: number[]) => weights.map((w) => e0.map((v, c) => ((64 - w) * v + w * e1[c]! + 32) >> 6));
    const decodeAll = (q: number[], p: number) => q.map((v, c) => bc7Decode(v, bitsOf(c), p));
    const quantAll = (e: number[], p: number) => e.map((v, c) => bc7Quantize(clamp(v, 0, 255), bitsOf(c), p));
    let best: BC7Subset | null = null;
    for (const [p0, p1] of pCombos) {
        const fit = fitEndpoints(points, n, weights.map((w) => w / 64), (e) => decodeAll(quantAll(e, p0), p0), palette, sq);
        // fitEndpoints snaps both endpoints with p0; requantize the second with its own p-bit.
        const q0 = quantAll(fit.e0, p0);
        const q1 = quantAll(fit.e1, p1);
        const pal = palette(decodeAll(q0, p0), decodeAll(q1, p1));
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
        if (!best || err < best.err) best = { q: [q0, q1], p: [p0, p1], indices, err };
    }
    return best!;
}

/** Residual of the subset's texels about their principal line (the partition-ranking estimate). */
function lineResidual(points: number[][], n: number): number {
    if (points.length < 2) return 0;
    const { mean, axis } = principalAxis(points, n);
    let r = 0;
    for (const p of points) {
        let d = 0;
        let len2 = 0;
        for (let c = 0; c < n; c++) {
            const x = p[c]! - mean[c]!;
            d += x * axis[c]!;
            len2 += x * x;
        }
        r += len2 - d * d;
    }
    return r;
}

interface BC7Candidate {
    m: BC7Mode;
    partition: number;
    subsets: BC7Subset[];
    err: number;
}

const kPartitionsTried = 4;

function encodeBC7Mode(texels: number[][], m: BC7Mode, opaque: boolean): BC7Candidate | null {
    const n = m.alphaBits > 0 ? 4 : 3;
    // RGB-only modes decode alpha as 255.
    if (n === 3 && !opaque) return null;
    const table = m.subsets === 2 ? kPartitions2 : m.subsets === 3 ? kPartitions3 : ["0000000000000000"];
    const count = m.subsets === 1 ? 1 : 1 << m.partitionBits;
    const groups = (part: number) => {
        const g: number[][][] = Array.from({ length: m.subsets }, () => []);
        for (let t = 0; t < 16; t++) g[table[part]!.charCodeAt(t) - 48]!.push(texels[t]!);
        return g;
    };
    const ranked = Array.from({ length: count }, (_, part) => ({ part, r: groups(part).reduce((s, g) => s + lineResidual(g, n), 0) }))
        .sort((a, b) => a.r - b.r)
        .slice(0, kPartitionsTried);
    let best: BC7Candidate | null = null;
    for (const { part } of ranked) {
        const subsets = groups(part).map((g) => bc7FitSubset(g, m, n));
        const err = subsets.reduce((s, x) => s + x.err, 0);
        if (!best || err < best.err) best = { m, partition: part, subsets, err };
    }
    return best;
}

function writeBC7(c: BC7Candidate, dst: Uint8Array): void {
    const { m, partition } = c;
    const table = m.subsets === 2 ? kPartitions2 : m.subsets === 3 ? kPartitions3 : ["0000000000000000"];
    const anchors = m.subsets === 2 ? [0, kAnchor2[partition]!] : m.subsets === 3 ? [0, ...kAnchor3[partition]!] : [0];
    // Per-texel index, with each subset's anchor forced to a 0 top bit by swapping its endpoints.
    const cursor = new Array<number>(m.subsets).fill(0);
    const indices = new Array<number>(16);
    const top = 1 << (m.indexBits - 1);
    const flip = c.subsets.map((s, k) => {
        const anchorSlot = table[partition]!.slice(0, anchors[k]!).split("").filter((d) => Number(d) === k).length;
        return s.indices[anchorSlot]! >= top;
    });
    for (let t = 0; t < 16; t++) {
        const k = table[partition]!.charCodeAt(t) - 48;
        const i = c.subsets[k]!.indices[cursor[k]!++]!;
        indices[t] = flip[k] ? (1 << m.indexBits) - 1 - i : i;
    }
    const ends = c.subsets.map((s, k) => (flip[k] ? { q: [s.q[1], s.q[0]], p: [s.p[1], s.p[0]] } : { q: s.q, p: s.p }));
    const w = new BitWriter(dst);
    w.write(1 << m.mode, m.mode + 1);
    w.write(partition, m.partitionBits);
    const channels = m.alphaBits > 0 ? 4 : 3;
    for (let ch = 0; ch < channels; ch++) {
        for (const e of ends) {
            w.write(e.q[0]![ch]!, ch < 3 ? m.colorBits : m.alphaBits);
            w.write(e.q[1]![ch]!, ch < 3 ? m.colorBits : m.alphaBits);
        }
    }
    if (m.pbits === "unique") for (const e of ends) e.p.forEach((p) => w.write(p, 1));
    else if (m.pbits === "shared") for (const e of ends) w.write(e.p[0]!, 1);
    for (let t = 0; t < 16; t++) w.write(indices[t]!, anchors.includes(t) ? m.indexBits - 1 : m.indexBits);
}

/** Nearest-palette indices of 1..4-channel points and their total squared error. */
function bc7Assign(points: number[][], pal: number[][]): { indices: number[]; err: number } {
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
    return { indices, err };
}

/**
 * Modes 4 and 5: one subset with color and alpha on separate index sets; `rotation` swaps
 * alpha with R/G/B (the decoder swaps back), mode 4's `idxMode` picks which gets 3-bit indices.
 */
function encodeBC7Separate(texels: number[][], mode: 4 | 5, rotation: number, idxMode: number): { err: number; write: (dst: Uint8Array) => void } {
    const rot = texels.map((p) => {
        const q = [...p];
        if (rotation > 0) [q[rotation - 1], q[3]] = [q[3]!, q[rotation - 1]!];
        return q;
    });
    const [colorBits, alphaBits] = mode === 4 ? [5, 6] : [7, 8];
    const colorW = mode === 4 && idxMode === 1 ? kWeights3 : kWeights2;
    const alphaW = mode === 4 && idxMode === 0 ? kWeights3 : kWeights2;
    const palette = (weights: number[]) => (e0: number[], e1: number[]) => weights.map((w) => e0.map((v, c) => ((64 - w) * v + w * e1[c]! + 32) >> 6));
    const fitPart = (points: number[][], bits: number, weights: number[]) => {
        const quant = (e: number[]) => e.map((v) => bc7Quantize(clamp(v, 0, 255), bits, -1));
        const decode = (q: number[]) => q.map((v) => bc7Decode(v, bits, -1));
        const fit = fitEndpoints(points, points[0]!.length, weights.map((w) => w / 64), (e) => decode(quant(e)), palette(weights), sq);
        let [q0, q1] = [quant(fit.e0), quant(fit.e1)];
        let { indices, err } = bc7Assign(points, palette(weights)(decode(q0), decode(q1)));
        // Texel 0 is the anchor of both index sets.
        if (indices[0]! >= weights.length / 2) {
            [q0, q1] = [q1, q0];
            indices = indices.map((i) => weights.length - 1 - i);
        }
        return { q0, q1, indices, err };
    };
    const color = fitPart(rot.map((p) => p.slice(0, 3)), colorBits, colorW);
    const alpha = fitPart(rot.map((p) => [p[3]!]), alphaBits, alphaW);
    return {
        err: color.err + alpha.err,
        write: (dst) => {
            const w = new BitWriter(dst);
            w.write(1 << mode, mode + 1);
            w.write(rotation, 2);
            if (mode === 4) w.write(idxMode, 1);
            for (let c = 0; c < 3; c++) {
                w.write(color.q0[c]!, colorBits);
                w.write(color.q1[c]!, colorBits);
            }
            w.write(alpha.q0[0]!, alphaBits);
            w.write(alpha.q1[0]!, alphaBits);
            // The 2-bit index set comes first.
            const [first, second] = mode === 4 && idxMode === 1 ? [alpha, color] : [color, alpha];
            const bitsOf = (set: typeof color) => ((set === color ? colorW : alphaW).length === 8 ? 3 : 2);
            for (const set of [first, second]) set.indices.forEach((i, t) => w.write(i, t === 0 ? bitsOf(set) - 1 : bitsOf(set)));
        },
    };
}

/** Tries every mode and keeps the one with the least squared error. */
function encodeBC7(b: Float32Array, dst: Uint8Array): void {
    const texels = Array.from({ length: 16 }, (_, t) => [0, 1, 2, 3].map((c) => clamp01(b[t * 4 + c]!) * 255));
    const opaque = texels.every((p) => p[3]! >= 254.5);
    let best: { err: number; write: (dst: Uint8Array) => void } | null = null;
    for (const m of kBC7Modes) {
        const c = encodeBC7Mode(opaque && m.alphaBits > 0 ? texels.map((p) => [p[0]!, p[1]!, p[2]!, 255]) : texels, m, opaque);
        if (c && (!best || c.err < best.err)) best = { err: c.err, write: (d) => writeBC7(c, d) };
    }
    for (const mode of [4, 5] as const) {
        for (let rotation = 0; rotation < 4; rotation++) {
            for (const idxMode of mode === 4 ? [0, 1] : [0]) {
                const c = encodeBC7Separate(texels, mode, rotation, idxMode);
                if (c.err < best!.err) best = c;
            }
        }
    }
    best!.write(dst);
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
