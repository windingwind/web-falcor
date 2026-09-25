/**
 * OpenVDB (.vdb) parser and NanoVDB (v32.3 ABI) grid builder — TS ports of
 * the validated reference implementations in scripts/vdb/ (vdb_parse.py,
 * vdb_to_nvdb.py; native-validated 0/500 point mismatches via Mogwai's
 * header-only NanoVDB loader).
 *
 * Scope mirrors what Falcor's volume assets need: FloatGrid with 5-4-3 tree,
 * file version >= 222, optionally half floats, and any combination of the
 * COMPRESS_ACTIVE_MASK / COMPRESS_ZIP / COMPRESS_BLOSC codecs (openvdb.org's
 * sample volumes are blosc+LZ4). Linear maps are supported where they reduce
 * to a uniform scale plus a translation, which is what NanoVDB stores.
 * The builder emits the exact in-memory buffer PNanoVDB.h traverses
 * (breadth-first layout, subtree stats, EMPTY checksum).
 */

import { RuntimeError } from "../../Core/Error.js";
import { bloscDecompress } from "../../Utils/Compression/Blosc.js";
import { inflate } from "../../Utils/Compression/Inflate.js";

export interface ParsedFloatGrid {
    translation: [number, number, number];
    scale: number;
    background: number;
    /** Leaf origins (x,y,z per leaf) in stream order. */
    leafOrigins: [number, number, number][];
    /** 64 B value mask per leaf (bit n = voxel n, x-major, little bit order). */
    leafMasks: Uint8Array[];
    /** 512 values per leaf (background where inactive). */
    leafValues: Float32Array[];
}

/**
 * NanoVDB's fog volumes (Grid::createSphere/createBox: nanovdb::createFogVolumeSphere/Box), built
 * like its GridBuilder so the values match native exactly, quirks included:
 * - the narrow-band signed distances (|v| < halfWidth voxels, world units) of initSphere/initBox;
 * - sdfToLevelSet's bottom-up scanline signedFloodFill over leaves (8^3), lower (16^3 leaves) and
 *   upper (32^3 lower) nodes, which can leave interior tiles "outside" when few band voxels
 *   reach a node (thin bands);
 * - sdfToFog: inside -> 1, band -> -v / (halfWidth * voxelSize), outside inactive.
 * Active tiles are expanded to leaves (the web NanoVDB writer emits leaves only).
 */
function buildFogVolume(sdf: (i: number, j: number, k: number) => number, lo: [number, number, number], hi: [number, number, number], voxelSize: number, halfWidth: number): ParsedFloatGrid {
    const f = Math.fround;
    const outside = f(halfWidth * voxelSize); // background
    const vs = f(voxelSize);
    type Leaf = { origin: [number, number, number]; values: Float32Array; mask: Uint8Array };
    type Lower = { origin: [number, number, number]; children: Map<number, Leaf>; tiles: Float32Array; active: Uint8Array };
    type Upper = { origin: [number, number, number]; children: Map<number, Lower>; tiles: Float32Array; active: Uint8Array };
    const uppers = new Map<string, Upper>();
    const key = (o: number[]) => o.join(",");
    const leafIndex = (i: number, j: number, k: number) => ((i & 7) << 6) | ((j & 7) << 3) | (k & 7); // NanoVDB order (z fastest)
    const lowerIndex = (i: number, j: number, k: number) => (((i & 127) >> 3) << 8) | (((j & 127) >> 3) << 4) | ((k & 127) >> 3);
    const upperIndex = (i: number, j: number, k: number) => (((i & 4095) >> 7) << 10) | (((j & 4095) >> 7) << 5) | ((k & 4095) >> 7);
    const setValue = (i: number, j: number, k: number, v: number) => {
        const uo: [number, number, number] = [i & ~4095, j & ~4095, k & ~4095];
        let up = uppers.get(key(uo));
        if (!up) uppers.set(key(uo), (up = { origin: uo, children: new Map(), tiles: new Float32Array(32768).fill(outside), active: new Uint8Array(32768) }));
        let low = up.children.get(upperIndex(i, j, k));
        if (!low) up.children.set(upperIndex(i, j, k), (low = { origin: [i & ~127, j & ~127, k & ~127], children: new Map(), tiles: new Float32Array(4096).fill(outside), active: new Uint8Array(4096) }));
        let leaf = low.children.get(lowerIndex(i, j, k));
        if (!leaf) low.children.set(lowerIndex(i, j, k), (leaf = { origin: [i & ~7, j & ~7, k & ~7], values: new Float32Array(512).fill(outside), mask: new Uint8Array(512) }));
        leaf.values[leafIndex(i, j, k)] = v;
        leaf.mask[leafIndex(i, j, k)] = 1;
    };
    // initSphere/initBox: narrow-band distances in world units.
    for (let i = lo[0]; i <= hi[0]; i++)
        for (let j = lo[1]; j <= hi[1]; j++)
            for (let k = lo[2]; k <= hi[2]; k++) {
                const v = sdf(i, j, k);
                if (Math.abs(v) < halfWidth) setValue(i, j, k, f(vs * v));
            }
    // signedFloodFill: the same scanline over a node's slots at every level (LOG2DIM 3/4/5).
    const floodFill = (log2: number, isOn: (n: number) => boolean, first: (n: number) => number, last: (n: number) => number, fill: (n: number, inside: boolean) => void) => {
        const size = 1 << (3 * log2);
        let start = -1;
        for (let n = 0; n < size; n++) if (isOn(n)) { start = n; break; }
        if (start < 0) return;
        let xInside = first(start) < 0;
        let yInside = xInside, zInside = xInside;
        const dim = 1 << log2;
        for (let x = 0; x < dim; x++) {
            const x00 = x << (2 * log2);
            if (isOn(x00)) xInside = last(x00) < 0;
            yInside = xInside;
            for (let y = 0; y < dim; y++) {
                const xy0 = x00 + (y << log2);
                if (isOn(xy0)) yInside = last(xy0) < 0;
                zInside = yInside;
                for (let z = 0; z < dim; z++) {
                    const xyz = xy0 + z;
                    if (isOn(xyz)) zInside = last(xyz) < 0;
                    else fill(xyz, zInside);
                }
            }
        }
    };
    const leafFirst = (l: Leaf) => l.values[0]!;
    const leafLast = (l: Leaf) => l.values[511]!;
    const lowerFirst = (n: Lower) => (n.children.has(0) ? leafFirst(n.children.get(0)!) : n.tiles[0]!);
    const lowerLast = (n: Lower) => (n.children.has(4095) ? leafLast(n.children.get(4095)!) : n.tiles[4095]!);
    const insideValue = -outside;
    for (const up of uppers.values())
        for (const low of up.children.values())
            for (const leaf of low.children.values())
                // Leaf: the values themselves; "on" = active voxel, and first == last == its own value.
                floodFill(3, (n) => leaf.mask[n] === 1, (n) => leaf.values[n]!, (n) => leaf.values[n]!, (n, inside) => (leaf.values[n] = inside ? insideValue : outside));
    for (const up of uppers.values())
        for (const low of up.children.values())
            floodFill(4, (n) => low.children.has(n), (n) => leafFirst(low.children.get(n)!), (n) => leafLast(low.children.get(n)!), (n, inside) => (low.tiles[n] = inside ? insideValue : outside));
    for (const up of uppers.values())
        floodFill(5, (n) => up.children.has(n), (n) => lowerFirst(up.children.get(n)!), (n) => lowerLast(up.children.get(n)!), (n, inside) => (up.tiles[n] = inside ? insideValue : outside));
    // sdfToFog on voxels and tiles.
    const w = f(1 / -outside);
    const fog = (v: number): [number, boolean] => (v > 0 ? [0, false] : [v > -outside ? f(v * w) : 1, true]);
    // Emit leaves in NanoVDB/OpenVDB order (z fastest), as buildNanoVDBGrid copies them; active tiles become full leaves of their value.
    const leafOrigins: [number, number, number][] = [];
    const leafMasks: Uint8Array[] = [];
    const leafValues: Float32Array[] = [];
    const emit = (origin: [number, number, number], value: (lx: number, ly: number, lz: number) => [number, boolean]) => {
        const values = new Float32Array(512);
        const mask = new Uint8Array(64);
        let any = false;
        for (let n = 0; n < 512; n++) {
            const [v, on] = value(n >> 6, (n >> 3) & 7, n & 7);
            if (!on) continue;
            values[n] = v;
            mask[n >> 3]! |= 1 << (n & 7);
            any = true;
        }
        if (any) {
            leafOrigins.push(origin);
            leafMasks.push(mask);
            leafValues.push(values);
        }
    };
    for (const up of uppers.values()) {
        for (const low of up.children.values()) {
            for (const leaf of low.children.values()) emit(leaf.origin, (x, y, z) => fog(leaf.values[(x << 6) | (y << 3) | z]!));
            for (let n = 0; n < 4096; n++) {
                if (low.children.has(n)) continue;
                const [v, on] = fog(low.tiles[n]!);
                if (on) emit([low.origin[0] + ((n >> 8) << 3), low.origin[1] + (((n >> 4) & 15) << 3), low.origin[2] + ((n & 15) << 3)], () => [v, true]);
            }
        }
        for (let n = 0; n < 32768; n++) {
            if (up.children.has(n)) continue;
            const [v, on] = fog(up.tiles[n]!);
            if (!on) continue;
            const o = [up.origin[0] + ((n >> 10) << 7), up.origin[1] + (((n >> 5) & 31) << 7), up.origin[2] + ((n & 31) << 7)];
            for (let a = 0; a < 128; a += 8) for (let b2 = 0; b2 < 128; b2 += 8) for (let c = 0; c < 128; c += 8) emit([o[0]! + a, o[1]! + b2, o[2]! + c], () => [v, true]);
        }
    }
    return { translation: [0, 0, 0], scale: voxelSize, background: 0, leafOrigins, leafMasks, leafValues };
}

const kEmptyGrid = (voxelSize: number): ParsedFloatGrid => ({ translation: [0, 0, 0], scale: voxelSize, background: 0, leafOrigins: [], leafMasks: [], leafValues: [] });

/** Mirrors Grid::createSphere (nanovdb::createFogVolumeSphere, centered at the origin). */
export function buildSphereGrid(radius: number, voxelSize: number, blendRange = 3): ParsedFloatGrid {
    const f = Math.fround;
    const r0 = f(f(radius) / f(voxelSize));
    const rmax = f(r0 + f(blendRange));
    if (r0 < 1.5) return kEmptyGrid(voxelSize); // below the Nyquist frequency
    const lo = Math.floor(-rmax), hi = Math.ceil(rmax);
    return buildFogVolume((i, j, k) => f(f(Math.sqrt(f(f(f(j * j) + f(i * i)) + f(k * k)))) - r0), [lo, lo, lo], [hi, hi, hi], voxelSize, blendRange);
}

/** Mirrors Grid::createBox (nanovdb::createFogVolumeBox, centered at the origin). */
export function buildBoxGrid(width: number, height: number, depth: number, voxelSize: number, blendRange = 3): ParsedFloatGrid {
    const f = Math.fround;
    const two = f(2 * f(voxelSize));
    const r = [f(f(width) / two), f(f(height) / two), f(f(depth) / two)];
    if (Math.min(...r) < 1.5) return kEmptyGrid(voxelSize);
    const lo = r.map((x) => Math.floor(f(-x - f(blendRange)))) as [number, number, number];
    const hi = r.map((x) => Math.ceil(f(x + f(blendRange)))) as [number, number, number];
    const pos = (x: number) => (x > 0 ? x : 0);
    const neg = (x: number) => (x < 0 ? x : 0);
    return buildFogVolume(
        (i, j, k) => {
            const q1 = f(Math.abs(i) - r[0]!), q2 = f(Math.abs(j) - r[1]!), q3 = f(Math.abs(k) - r[2]!);
            const x2y2 = f(f(pos(q1) * pos(q1)) + f(pos(q2) * pos(q2)));
            return f(f(Math.sqrt(f(x2y2 + f(pos(q3) * pos(q3))))) + neg(Math.max(Math.max(q1, q2), q3)));
        },
        lo,
        hi,
        voxelSize,
        blendRange,
    );
}

/** NanoVDB grid statistics and point lookup over a ParsedFloatGrid (python Grid readback). */
export function parsedGridStats(g: ParsedFloatGrid): { voxelCount: number; minIndex: [number, number, number]; maxIndex: [number, number, number]; minValue: number; maxValue: number } {
    let voxelCount = 0, minValue = Infinity, maxValue = -Infinity;
    const minIndex: [number, number, number] = [Infinity, Infinity, Infinity];
    const maxIndex: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    g.leafOrigins.forEach(([lx, ly, lz], l) => {
        const mask = g.leafMasks[l]!, values = g.leafValues[l]!;
        for (let n = 0; n < 512; n++) {
            if (!(mask[n >> 3]! & (1 << (n & 7)))) continue;
            const p = [lx + (n >> 6), ly + ((n >> 3) & 7), lz + (n & 7)];
            voxelCount++;
            for (let c = 0; c < 3; c++) {
                minIndex[c] = Math.min(minIndex[c]!, p[c]!);
                maxIndex[c] = Math.max(maxIndex[c]!, p[c]!);
            }
            minValue = Math.min(minValue, values[n]!);
            maxValue = Math.max(maxValue, values[n]!);
        }
    });
    if (voxelCount === 0) return { voxelCount, minIndex: [0, 0, 0], maxIndex: [0, 0, 0], minValue: 0, maxValue: 0 };
    return { voxelCount, minIndex, maxIndex, minValue, maxValue };
}

/** The value at index-space voxel (i, j, k); the background where no voxel is stored. */
export function parsedGridValue(g: ParsedFloatGrid, i: number, j: number, k: number): number {
    const [lx, ly, lz] = [Math.floor(i / 8) * 8, Math.floor(j / 8) * 8, Math.floor(k / 8) * 8];
    const l = g.leafOrigins.findIndex((o) => o[0] === lx && o[1] === ly && o[2] === lz);
    if (l < 0) return g.background;
    return g.leafValues[l]![((i - lx) << 6) | ((j - ly) << 3) | (k - lz)]!;
}

function halfToFloat(h: number): number {
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity;
    return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

class Reader {
    readonly view: DataView;
    o = 0;
    constructor(readonly data: Uint8Array, offset = 0) {
        this.view = new DataView(data.buffer, data.byteOffset, data.byteLength);
        this.o = offset;
    }
    u32(): number { const v = this.view.getUint32(this.o, true); this.o += 4; return v; }
    i32(): number { const v = this.view.getInt32(this.o, true); this.o += 4; return v; }
    i64(): number { const v = Number(this.view.getBigInt64(this.o, true)); this.o += 8; return v; }
    f32(): number { const v = this.view.getFloat32(this.o, true); this.o += 4; return v; }
    f64(): number { const v = this.view.getFloat64(this.o, true); this.o += 8; return v; }
    byte(): number { return this.data[this.o++]!; }
    raw(n: number): Uint8Array { const v = this.data.subarray(this.o, this.o + n); this.o += n; return v; }
    str(): string { return new TextDecoder().decode(this.raw(this.u32())); }
    coord(): [number, number, number] { return [this.i32(), this.i32(), this.i32()]; }
    half(): number { const v = this.view.getUint16(this.o, true); this.o += 2; return halfToFloat(v); }
}

const bit = (mask: Uint8Array, n: number): boolean => (mask[n >> 3]! & (1 << (n & 7))) !== 0;

function popcount(mask: Uint8Array): number {
    let c = 0;
    for (let i = 0; i < mask.length; i++) {
        let b = mask[i]!;
        while (b) { c += b & 1; b >>= 1; }
    }
    return c;
}

/** Value codecs a grid's data may use (io::Compression flags). */
interface ValueCodec {
    /** COMPRESS_ACTIVE_MASK: only active voxels are stored, inactive ones are implied. */
    maskCompressed: boolean;
    /** Stream codec wrapping the value bytes. */
    stream: "none" | "zip" | "blosc";
}

/**
 * Mirrors io::readData: raw bytes, or a codec chunk prefixed by its int64 size.
 * A size <= 0 means the writer stored the data uncompressed after all, and its
 * magnitude is the uncompressed byte count.
 */
function readValueBytes(r: Reader, byteCount: number, codec: ValueCodec): Uint8Array {
    if (codec.stream === "none") return r.raw(byteCount);
    const size = r.i64();
    if (size <= 0) return r.raw(-size);
    const payload = r.raw(size);
    return codec.stream === "blosc" ? bloscDecompress(payload) : inflate(payload, byteCount);
}

/** Decodes `count` values from raw bytes (half or single precision). */
function decodeValues(bytes: Uint8Array, count: number, half: boolean): Float32Array {
    const out = new Float32Array(count);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let i = 0; i < count; i++) out[i] = half ? halfToFloat(view.getUint16(i * 2, true)) : view.getFloat32(i * 4, true);
    return out;
}

/** Mirrors io::readCompressedValues (all metadata cases, all codecs). */
function readCompressed(r: Reader, count: number, valueMask: Uint8Array, half: boolean, background: number, codec: ValueCodec): Float32Array {
    const meta = r.byte();
    if (meta >= 7) throw new RuntimeError(`OpenVDB: bad compression metadata byte ${meta}`);

    // Inactive values: implied by the metadata, or stored at FULL precision
    // (sizeof(ValueType)) even in half grids — only the buffer is halved.
    let inactive0 = meta === 0 ? background : -background;
    let inactive1 = background;
    if (meta === 2 || meta === 4 || meta === 5) {
        inactive0 = r.f32();
        if (meta === 5) inactive1 = r.f32();
    }
    const selectionMask = meta === 3 || meta === 4 || meta === 5 ? r.raw(count / 8) : null;

    // With mask compression only the active voxels are stored (NO_MASK_AND_ALL_VALS excepted).
    const storedCount = codec.maskCompressed && meta !== 6 ? popcount(valueMask) : count;
    // HalfReader::read returns early on an empty buffer, so nothing at all is
    // written — the plain path still emits the codec's size prefix. Verified
    // against openvdb.org's own half (cube) and float (torus) samples.
    const bytes = half && storedCount === 0 ? new Uint8Array(0) : readValueBytes(r, storedCount * (half ? 2 : 4), codec);
    const values = decodeValues(bytes, storedCount, half);
    if (storedCount === count) return values;

    const out = new Float32Array(count);
    let n = 0;
    for (let i = 0; i < count; i++) {
        out[i] = bit(valueMask, i) ? values[n++]! : selectionMask !== null && bit(selectionMask, i) ? inactive1 : inactive0;
    }
    return out;
}

/**
 * Doubles each linear map type writes (math::MapRegistry) and how to reduce it
 * to NanoVDB's uniform scale + translation.
 */
const kMapLayouts: Record<string, { doubles: number; translationAt: number | null; scaleAt: number | null }> = {
    // ScaleMap: scale, voxelSize, scaleInverse, invScaleSqr, invTwiceScale.
    UniformScaleMap: { doubles: 15, translationAt: null, scaleAt: 0 },
    ScaleMap: { doubles: 15, translationAt: null, scaleAt: 0 },
    // ScaleTranslateMap: translation first, then the ScaleMap block.
    UniformScaleTranslateMap: { doubles: 18, translationAt: 0, scaleAt: 3 },
    ScaleTranslateMap: { doubles: 18, translationAt: 0, scaleAt: 3 },
    TranslationMap: { doubles: 3, translationAt: 0, scaleAt: null },
};

/** Reads the grid's transform; throws for maps NanoVDB's uniform-scale grid can't express. */
function readMap(g: Reader): { translation: [number, number, number]; scale: number } {
    const mapType = g.str();
    if (mapType === "AffineMap") {
        // Row-major 4x4; supported when it is a uniform scale with translation.
        const m: number[] = [];
        for (let i = 0; i < 16; i++) m.push(g.f64());
        const offDiagonal = [m[1], m[2], m[4], m[6], m[8], m[9]].some((v) => Math.abs(v!) > 1e-12);
        if (offDiagonal || Math.abs(m[0]! - m[5]!) > 1e-12 || Math.abs(m[0]! - m[10]!) > 1e-12) {
            throw new RuntimeError("OpenVDB: AffineMap with rotation/shear or non-uniform scale is unsupported");
        }
        return { translation: [m[12]!, m[13]!, m[14]!], scale: m[0]! };
    }
    const layout = kMapLayouts[mapType];
    if (!layout) throw new RuntimeError(`OpenVDB: unsupported map ${mapType}`);
    const d: number[] = [];
    for (let i = 0; i < layout.doubles; i++) d.push(g.f64());
    const translation: [number, number, number] = layout.translationAt === null ? [0, 0, 0] : [d[layout.translationAt]!, d[layout.translationAt + 1]!, d[layout.translationAt + 2]!];
    let scale = 1;
    if (layout.scaleAt !== null) {
        scale = d[layout.scaleAt]!;
        const sy = d[layout.scaleAt + 1]!;
        const sz = d[layout.scaleAt + 2]!;
        if (Math.abs(scale - sy) > 1e-12 || Math.abs(scale - sz) > 1e-12) {
            throw new RuntimeError(`OpenVDB: non-uniform voxel size (${scale}, ${sy}, ${sz}) is unsupported`);
        }
    }
    return { translation, scale };
}

/** Parses a FloatGrid (5-4-3 tree) from an OpenVDB .vdb file. */
export function parseOpenVDBFloatGrid(buffer: ArrayBuffer, gridname = "density"): ParsedFloatGrid {
    const data = new Uint8Array(buffer);
    const r = new Reader(data);
    if (r.i64() !== 0x56444220) throw new RuntimeError("Not an OpenVDB file");
    const fileVersion = r.u32();
    if (fileVersion < 222) throw new RuntimeError(`Unsupported OpenVDB file version ${fileVersion}`);
    r.u32(); r.u32(); // library version
    if (!r.byte()) throw new RuntimeError("OpenVDB: file without grid offsets unsupported");
    r.raw(36); // uuid
    for (let i = r.u32(); i > 0; i--) { r.str(); r.str(); r.raw(r.u32()); }

    let found: { type: string; gridPos: number; blockPos: number; endPos: number } | null = null;
    for (let i = r.u32(); i > 0; i--) {
        const name = r.str().split("\x1e")[0]!;
        const gtype = r.str();
        r.str(); // instance parent
        const gridPos = r.i64(), blockPos = r.i64(), endPos = r.i64();
        if (name === gridname) found = { type: gtype, gridPos, blockPos, endPos };
    }
    if (!found) throw new RuntimeError(`OpenVDB: grid '${gridname}' not found`);
    const half = found.type.endsWith("_HalfFloat");
    const base = half ? found.type.slice(0, -"_HalfFloat".length) : found.type;
    if (base !== "Tree_float_5_4_3") throw new RuntimeError(`OpenVDB: unsupported grid type ${found.type}`);

    const g = new Reader(data, found.gridPos);
    // io::Compression flags: 0x1 ZIP, 0x2 ACTIVE_MASK, 0x4 BLOSC.
    const compression = g.u32();
    if (compression & ~0x7) throw new RuntimeError(`OpenVDB: unknown compression flags ${compression}`);
    if ((compression & 0x1) && (compression & 0x4)) throw new RuntimeError("OpenVDB: both ZIP and BLOSC set");
    const codec: ValueCodec = {
        maskCompressed: (compression & 0x2) !== 0,
        stream: compression & 0x4 ? "blosc" : compression & 0x1 ? "zip" : "none",
    };
    for (let i = g.u32(); i > 0; i--) { g.str(); g.str(); g.raw(g.u32()); }
    const { translation, scale } = readMap(g);

    if (g.u32() !== 1) throw new RuntimeError("OpenVDB: unexpected tree buffer count");
    const background = g.f32();
    const numTiles = g.u32();
    const numChildren = g.u32();
    // Root tiles are constant regions at the top level; NanoVDB's builder here
    // emits leaf nodes only, so a file that uses them would lose data.
    if (numTiles !== 0) throw new RuntimeError(`OpenVDB: root tiles are unsupported (${numTiles} in this grid)`);

    const leafOrigins: [number, number, number][] = [];
    const leafMasks: Uint8Array[] = [];
    for (let c = 0; c < numChildren; c++) {
        const org5 = g.coord();
        const cm5 = g.raw(4096).slice();
        const vm5 = g.raw(4096).slice();
        readCompressed(g, 32768, vm5, half, background, codec);
        for (let i5 = 0; i5 < 32768; i5++) {
            if (!bit(cm5, i5)) continue;
            const org4: [number, number, number] = [
                org5[0] + ((i5 >> 10) << 7),
                org5[1] + (((i5 >> 5) & 31) << 7),
                org5[2] + ((i5 & 31) << 7),
            ];
            const cm4 = g.raw(512).slice();
            const vm4 = g.raw(512).slice();
            readCompressed(g, 4096, vm4, half, background, codec);
            for (let i4 = 0; i4 < 4096; i4++) {
                if (!bit(cm4, i4)) continue;
                leafOrigins.push([
                    org4[0] + ((i4 >> 8) << 3),
                    org4[1] + (((i4 >> 4) & 15) << 3),
                    org4[2] + ((i4 & 15) << 3),
                ]);
                leafMasks.push(g.raw(64).slice());
            }
        }
    }
    if (g.o !== found.blockPos) throw new RuntimeError(`OpenVDB: topology ended at ${g.o}, expected ${found.blockPos}`);

    const leafValues: Float32Array[] = [];
    for (let i = 0; i < leafOrigins.length; i++) {
        const vm = g.raw(64);
        for (let b = 0; b < 64; b++) {
            if (vm[b] !== leafMasks[i]![b]) throw new RuntimeError("OpenVDB: buffer mask mismatch");
        }
        leafValues.push(readCompressed(g, 512, leafMasks[i]!, half, background, codec));
    }
    if (g.o !== found.endPos && g.o !== data.length) throw new RuntimeError(`OpenVDB: buffers ended at ${g.o}`);

    return { translation, scale, background, leafOrigins, leafMasks, leafValues };
}

// ---------------------------------------------------------------------------
// NanoVDB v32.3 builder (see scripts/vdb/vdb_to_nvdb.py for the layout notes).
// ---------------------------------------------------------------------------

const GRID_DATA_SIZE = 672;
const TREE_DATA_SIZE = 64;
const ROOT_DATA_SIZE = 64;
const ROOT_TILE_SIZE = 32;
const UPPER_SIZE = 8256 + 32768 * 8;
const LOWER_SIZE = 1088 + 4096 * 8;
const LEAF_SIZE = 96 + 512 * 4;

const NANOVDB_MAGIC = 0x304244566f6e614en;
const NANOVDB_VERSION = (32 << 21) | (3 << 10) | 3;

interface SubtreeStats {
    min: number; max: number; avg: number; std: number;
    bbmin: [number, number, number]; bbmax: [number, number, number];
    count: number; sum: number; sum2: number;
}

function keyOf(o: [number, number, number]): string {
    return `${o[0]},${o[1]},${o[2]}`;
}

function cmpOrigin(a: [number, number, number], b: [number, number, number]): number {
    return a[0] - b[0] || a[1] - b[1] || a[2] - b[2];
}

/** Builds the NanoVDB v32.3 grid buffer PNanoVDB traverses. */
export function buildNanoVDBGrid(grid: ParsedFloatGrid, gridname = "density"): Uint8Array {
    const { leafOrigins, leafMasks, leafValues, translation, scale, background } = grid;

    // Tree structure (Maps keep python-dict insertion order semantics).
    const uppers = new Map<string, { origin: [number, number, number]; children: Map<number, [number, number, number]> }>();
    const lowers = new Map<string, { origin: [number, number, number]; children: Map<number, number> }>(); // bit -> leaf index
    for (let i = 0; i < leafOrigins.length; i++) {
        const org = leafOrigins[i]!;
        const o4: [number, number, number] = [org[0] & ~127, org[1] & ~127, org[2] & ~127];
        const o5: [number, number, number] = [org[0] & ~4095, org[1] & ~4095, org[2] & ~4095];
        const b4 = (((org[0] & 127) >> 3) << 8) | (((org[1] & 127) >> 3) << 4) | ((org[2] & 127) >> 3);
        const b5 = (((o4[0] & 4095) >> 7) << 10) | (((o4[1] & 4095) >> 7) << 5) | ((o4[2] & 4095) >> 7);
        let u = uppers.get(keyOf(o5));
        if (!u) uppers.set(keyOf(o5), (u = { origin: o5, children: new Map() }));
        u.children.set(b5, o4);
        let l = lowers.get(keyOf(o4));
        if (!l) lowers.set(keyOf(o4), (l = { origin: o4, children: new Map() }));
        l.children.set(b4, i);
    }
    const upperList = [...uppers.values()].sort((a, b) => cmpOrigin(a.origin, b.origin));
    const lowerList = [...lowers.values()].sort((a, b) => cmpOrigin(a.origin, b.origin));
    const nUpper = upperList.length, nLower = lowerList.length, nLeaf = leafOrigins.length;

    // Leaves grouped by lower node, sorted bit order (breadth-first layout).
    const leafList: number[] = [];
    for (const l of lowerList) {
        for (const b of [...l.children.keys()].sort((a, b2) => a - b2)) leafList.push(l.children.get(b)!);
    }

    // Per-leaf stats (sequential f64 summation, mirroring the reference tool).
    const leafStats: SubtreeStats[] = new Array(nLeaf);
    let totalActive = 0;
    for (const li of leafList) {
        const vm = leafMasks[li]!, vals = leafValues[li]!, org = leafOrigins[li]!;
        let mn = Infinity, mx = -Infinity, s = 0, s2 = 0, cnt = 0;
        let x0 = 8, y0 = 8, z0 = 8, x1 = -1, y1 = -1, z1 = -1;
        for (let n = 0; n < 512; n++) {
            if (!bit(vm, n)) continue;
            const v = vals[n]!;
            mn = Math.min(mn, v); mx = Math.max(mx, v);
            s += v; s2 += v * v; cnt++;
            const x = n >> 6, y = (n >> 3) & 7, z = n & 7;
            x0 = Math.min(x0, x); y0 = Math.min(y0, y); z0 = Math.min(z0, z);
            x1 = Math.max(x1, x); y1 = Math.max(y1, y); z1 = Math.max(z1, z);
        }
        const avg = s / cnt;
        leafStats[li] = {
            min: mn, max: mx, avg, std: Math.sqrt(Math.max(0, s2 / cnt - avg * avg)),
            bbmin: [org[0] + x0, org[1] + y0, org[2] + z0],
            bbmax: [org[0] + x1, org[1] + y1, org[2] + z1],
            count: cnt, sum: s, sum2: s2,
        };
        totalActive += cnt;
    }

    const agg = (children: SubtreeStats[]): SubtreeStats => {
        let cnt = 0, s = 0, s2 = 0, mn = Infinity, mx = -Infinity;
        const bbmin: [number, number, number] = [Infinity, Infinity, Infinity];
        const bbmax: [number, number, number] = [-Infinity, -Infinity, -Infinity];
        for (const c of children) {
            cnt += c.count; s += c.sum; s2 += c.sum2;
            mn = Math.min(mn, c.min); mx = Math.max(mx, c.max);
            for (let k = 0; k < 3; k++) {
                bbmin[k] = Math.min(bbmin[k]!, c.bbmin[k]!);
                bbmax[k] = Math.max(bbmax[k]!, c.bbmax[k]!);
            }
        }
        const avg = s / cnt;
        return { min: mn, max: mx, avg, std: Math.sqrt(Math.max(0, s2 / cnt - avg * avg)), bbmin, bbmax, count: cnt, sum: s, sum2: s2 };
    };
    const lowerStats = new Map<string, SubtreeStats>();
    for (const l of lowerList) lowerStats.set(keyOf(l.origin), agg([...l.children.values()].map((li) => leafStats[li]!)));
    const upperStats = new Map<string, SubtreeStats>();
    for (const u of upperList) upperStats.set(keyOf(u.origin), agg([...u.children.values()].map((o4) => lowerStats.get(keyOf(o4))!)));
    const rootStats = agg(upperList.map((u) => upperStats.get(keyOf(u.origin))!));

    // Offsets: grid | tree | root+tiles | upper | lower | leaf.
    const offTree = GRID_DATA_SIZE;
    const offRoot = offTree + TREE_DATA_SIZE;
    const offUpper0 = offRoot + ROOT_DATA_SIZE + nUpper * ROOT_TILE_SIZE;
    const offLower0 = offUpper0 + nUpper * UPPER_SIZE;
    const offLeaf0 = offLower0 + nLower * LOWER_SIZE;
    const gridSize = offLeaf0 + nLeaf * LEAF_SIZE;
    const upperOff = new Map(upperList.map((u, i) => [keyOf(u.origin), offUpper0 + i * UPPER_SIZE]));
    const lowerOff = new Map(lowerList.map((l, i) => [keyOf(l.origin), offLower0 + i * LOWER_SIZE]));
    const leafOff = new Map(leafList.map((li, i) => [li, offLeaf0 + i * LEAF_SIZE]));

    const buf = new Uint8Array(gridSize);
    const view = new DataView(buf.buffer);

    // GridData.
    view.setBigUint64(0, NANOVDB_MAGIC, true);
    view.setBigUint64(8, 0xffffffffffffffffn, true); // checksum EMPTY
    view.setUint32(16, NANOVDB_VERSION, true);
    view.setUint32(20, 2 | 4 | 8 | 16 | 32, true); // HasBBox|HasMinMax|HasAverage|HasStdDeviation|IsBreadthFirst
    view.setUint32(24, 0, true);
    view.setUint32(28, 1, true);
    view.setBigUint64(32, BigInt(gridSize), true);
    buf.set(new TextEncoder().encode(gridname), 40);
    let o = 40 + 256;
    const matf = [scale, 0, 0, 0, scale, 0, 0, 0, scale];
    const invf = [1 / scale, 0, 0, 0, 1 / scale, 0, 0, 0, 1 / scale];
    for (const v of matf) { view.setFloat32(o, v, true); o += 4; }
    for (const v of invf) { view.setFloat32(o, v, true); o += 4; }
    for (const v of translation) { view.setFloat32(o, v, true); o += 4; }
    view.setFloat32(o, 1, true); o += 4;
    for (const v of matf) { view.setFloat64(o, v, true); o += 8; }
    for (const v of invf) { view.setFloat64(o, v, true); o += 8; }
    for (const v of translation) { view.setFloat64(o, v, true); o += 8; }
    view.setFloat64(o, 1, true); o += 8;
    for (let k = 0; k < 3; k++) { view.setFloat64(o, rootStats.bbmin[k]! * scale + translation[k]!, true); o += 8; }
    for (let k = 0; k < 3; k++) { view.setFloat64(o, (rootStats.bbmax[k]! + 1) * scale + translation[k]!, true); o += 8; }
    for (let k = 0; k < 3; k++) { view.setFloat64(o, scale, true); o += 8; }
    view.setUint32(o, 2, true); o += 4;  // GridClass::FogVolume
    view.setUint32(o, 1, true); o += 4;  // GridType::Float
    view.setBigInt64(o, 0n, true); o += 8;
    view.setUint32(o, 0, true);

    // TreeData (offsets relative to tree start).
    view.setBigUint64(offTree, BigInt(offLeaf0 - offTree), true);
    view.setBigUint64(offTree + 8, BigInt(offLower0 - offTree), true);
    view.setBigUint64(offTree + 16, BigInt(offUpper0 - offTree), true);
    view.setBigUint64(offTree + 24, BigInt(offRoot - offTree), true);
    view.setUint32(offTree + 32, nLeaf, true);
    view.setUint32(offTree + 36, nLower, true);
    view.setUint32(offTree + 40, nUpper, true);
    view.setBigUint64(offTree + 56, BigInt(totalActive), true);

    // RootData + tiles.
    for (let k = 0; k < 3; k++) view.setInt32(offRoot + k * 4, rootStats.bbmin[k]!, true);
    for (let k = 0; k < 3; k++) view.setInt32(offRoot + 12 + k * 4, rootStats.bbmax[k]!, true);
    view.setUint32(offRoot + 24, nUpper, true);
    view.setFloat32(offRoot + 28, background, true);
    view.setFloat32(offRoot + 32, rootStats.min, true);
    view.setFloat32(offRoot + 36, rootStats.max, true);
    view.setFloat32(offRoot + 40, rootStats.avg, true);
    view.setFloat32(offRoot + 44, rootStats.std, true);
    upperList.forEach((u, i) => {
        // CoordToKey: uint32(ijk) >> 12, a logical shift (JS >> would sign-extend negative origins).
        const key = BigInt(u.origin[2] >>> 12) | (BigInt(u.origin[1] >>> 12) << 21n) | (BigInt(u.origin[0] >>> 12) << 42n);
        const to = offRoot + ROOT_DATA_SIZE + i * ROOT_TILE_SIZE;
        view.setBigUint64(to, key, true);
        view.setBigInt64(to + 8, BigInt(upperOff.get(keyOf(u.origin))! - offRoot), true);
        view.setUint32(to + 16, 0, true); // state: inactive (this tile has a child)
        view.setFloat32(to + 20, background, true);
    });

    // Internal nodes.
    const writeInternal = (
        base: number, st: SubtreeStats, maskBytes: number, childBits: number[],
        tableOff: number, childOffsets: Map<number, number>,
    ) => {
        // Each table slot is a union: a child offset where the child mask is set,
        // otherwise the region's tile value. Slots without a child cover empty
        // space, which reads back as the grid background (zero only happens to be
        // right for fog volumes; a level set's background is its narrow-band width).
        for (let b = 0; b < maskBytes * 8; b++) view.setFloat32(base + tableOff + b * 8, background, true);
        for (let k = 0; k < 3; k++) view.setInt32(base + k * 4, st.bbmin[k]!, true);
        for (let k = 0; k < 3; k++) view.setInt32(base + 12 + k * 4, st.bbmax[k]!, true);
        view.setBigUint64(base + 24, 0n, true);
        for (const b of childBits) buf[base + 32 + maskBytes + (b >> 3)] = buf[base + 32 + maskBytes + (b >> 3)]! | (1 << (b & 7));
        const statsOff = base + 32 + 2 * maskBytes;
        view.setFloat32(statsOff, st.min, true);
        view.setFloat32(statsOff + 4, st.max, true);
        view.setFloat32(statsOff + 8, st.avg, true);
        view.setFloat32(statsOff + 12, st.std, true);
        for (const [b, off] of childOffsets) view.setBigInt64(base + tableOff + b * 8, BigInt(off - base), true);
    };
    for (const u of upperList) {
        const base = upperOff.get(keyOf(u.origin))!;
        const offs = new Map<number, number>();
        for (const [b, o4] of u.children) offs.set(b, lowerOff.get(keyOf(o4))!);
        writeInternal(base, upperStats.get(keyOf(u.origin))!, 4096, [...u.children.keys()], 8256, offs);
    }
    for (const l of lowerList) {
        const base = lowerOff.get(keyOf(l.origin))!;
        const offs = new Map<number, number>();
        for (const [b, li] of l.children) offs.set(b, leafOff.get(li)!);
        writeInternal(base, lowerStats.get(keyOf(l.origin))!, 512, [...l.children.keys()], 1088, offs);
    }

    // Leaves.
    for (const li of leafList) {
        const base = leafOff.get(li)!;
        const st = leafStats[li]!;
        for (let k = 0; k < 3; k++) view.setInt32(base + k * 4, st.bbmin[k]!, true);
        for (let k = 0; k < 3; k++) buf[base + 12 + k] = st.bbmax[k]! - st.bbmin[k]!;
        buf[base + 15] = 0;
        buf.set(leafMasks[li]!, base + 16);
        view.setFloat32(base + 80, st.min, true);
        view.setFloat32(base + 84, st.max, true);
        view.setFloat32(base + 88, st.avg, true);
        view.setFloat32(base + 92, st.std, true);
        buf.set(new Uint8Array(leafValues[li]!.buffer, leafValues[li]!.byteOffset, 2048), base + 96);
    }

    return buf;
}

/** Extracts the raw grid buffer for `gridname` from a .nvdb file (codec NONE). */
export function extractGridFromNVDB(buffer: ArrayBuffer, gridname = "density"): Uint8Array {
    const data = new Uint8Array(buffer);
    const r = new Reader(data);
    if (r.view.getBigUint64(0, true) !== NANOVDB_MAGIC) throw new RuntimeError("Not a NanoVDB file");
    r.o = 8;
    r.u32(); // version
    const gridCount = r.view.getUint16(r.o, true); r.o += 2;
    const codec = r.view.getUint16(r.o, true); r.o += 2;
    if (codec !== 0) throw new RuntimeError("NanoVDB: compressed codecs unsupported");
    for (let i = 0; i < gridCount; i++) {
        const metaStart = r.o; // MetaData is 176 bytes
        const gridBytes = r.i64();
        const fileBytes = r.i64();
        r.o = metaStart + 32 + 4 + 4 + 48 + 24 + 24; // nameSize field
        const nameSize = r.u32();
        r.o = metaStart + 176;
        const name = new TextDecoder().decode(r.raw(nameSize)).replace(/\0+$/, "");
        if (name === gridname) return r.raw(gridBytes).slice();
        r.o += fileBytes;
    }
    throw new RuntimeError(`NanoVDB: grid '${gridname}' not found`);
}
