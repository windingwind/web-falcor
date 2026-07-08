/**
 * OpenVDB (.vdb) parser and NanoVDB (v32.3 ABI) grid builder — TS ports of
 * the validated reference implementations in tools/vdb/ (vdb_parse.py,
 * vdb_to_nvdb.py; native-validated 0/500 point mismatches via Mogwai's
 * header-only NanoVDB loader).
 *
 * Scope mirrors what Falcor's volume assets need: FloatGrid with 5-4-3 tree,
 * file version >= 222, COMPRESS_ACTIVE_MASK (optionally half floats),
 * UniformScaleTranslateMap. The builder emits the exact in-memory buffer
 * PNanoVDB.h traverses (breadth-first layout, subtree stats, EMPTY checksum).
 */

import { RuntimeError } from "../../Core/Error.js";

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

/** Builds a ParsedFloatGrid by sampling a density function over a world-space AABB
 *  at `voxelSize` resolution (for TriangleMesh-free procedural volumes). */
function buildProceduralGrid(
    density: (wx: number, wy: number, wz: number) => number,
    minW: [number, number, number],
    maxW: [number, number, number],
    voxelSize: number,
): ParsedFloatGrid {
    const leafFloor = (w: number) => Math.floor(Math.floor(w / voxelSize) / 8) * 8;
    const idxCeil = (w: number) => Math.ceil(w / voxelSize);
    const leafOrigins: [number, number, number][] = [];
    const leafMasks: Uint8Array[] = [];
    const leafValues: Float32Array[] = [];
    for (let lz = leafFloor(minW[2]); lz <= idxCeil(maxW[2]); lz += 8)
        for (let ly = leafFloor(minW[1]); ly <= idxCeil(maxW[1]); ly += 8)
            for (let lx = leafFloor(minW[0]); lx <= idxCeil(maxW[0]); lx += 8) {
                const values = new Float32Array(512);
                const mask = new Uint8Array(64);
                let any = false;
                for (let n = 0; n < 512; n++) {
                    const ix = lx + (n & 7);
                    const iy = ly + ((n >> 3) & 7);
                    const iz = lz + ((n >> 6) & 7);
                    const d = density(ix * voxelSize, iy * voxelSize, iz * voxelSize);
                    if (d > 0) {
                        values[n] = d;
                        mask[n >> 3]! |= 1 << (n & 7);
                        any = true;
                    }
                }
                if (any) {
                    leafOrigins.push([lx, ly, lz]);
                    leafMasks.push(mask);
                    leafValues.push(values);
                }
            }
    return { translation: [0, 0, 0], scale: voxelSize, background: 0, leafOrigins, leafMasks, leafValues };
}

/** Procedural unit-density sphere (mirrors Grid::createSphere). */
export function buildSphereGrid(radius: number, voxelSize: number): ParsedFloatGrid {
    const r2 = radius * radius;
    return buildProceduralGrid((x, y, z) => (x * x + y * y + z * z <= r2 ? 1 : 0), [-radius, -radius, -radius], [radius, radius, radius], voxelSize);
}

/** Procedural unit-density box (mirrors Grid::createBox). */
export function buildBoxGrid(width: number, height: number, depth: number, voxelSize: number): ParsedFloatGrid {
    const hx = width / 2;
    const hy = height / 2;
    const hz = depth / 2;
    return buildProceduralGrid((x, y, z) => (Math.abs(x) <= hx && Math.abs(y) <= hy && Math.abs(z) <= hz ? 1 : 0), [-hx, -hy, -hz], [hx, hy, hz], voxelSize);
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

/** io::readCompressedValues with COMPRESS_ACTIVE_MASK (no ZIP/BLOSC). */
function readCompressed(r: Reader, count: number, valueMask: Uint8Array, half: boolean, background: number): Float32Array {
    const meta = r.byte();
    if (meta >= 7) throw new RuntimeError(`OpenVDB: bad compression metadata byte ${meta}`);
    const rd = () => (half ? r.half() : r.f32());
    let inactive0: number | null = null;
    if (meta === 2 || meta === 4) inactive0 = rd();
    else if (meta === 5) { inactive0 = rd(); rd(); }
    let selMask: Uint8Array | null = null;
    if (meta === 3 || meta === 4 || meta === 5) selMask = r.raw(count / 8);
    const out = new Float32Array(count).fill(background);
    if (meta === 6) {
        for (let i = 0; i < count; i++) out[i] = rd();
        return out;
    }
    for (let i = 0; i < count; i++) {
        const active = bit(valueMask, i);
        const selected = selMask !== null && bit(selMask, i);
        if (active && !selected) out[i] = rd();
        else if (active && selected && inactive0 !== null) out[i] = inactive0;
    }
    return out;
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
    const compression = g.u32();
    if (compression !== 2) throw new RuntimeError(`OpenVDB: only ACTIVE_MASK compression supported (flags ${compression})`);
    for (let i = g.u32(); i > 0; i--) { g.str(); g.str(); g.raw(g.u32()); }
    const mapType = g.str();
    if (mapType !== "UniformScaleTranslateMap") throw new RuntimeError(`OpenVDB: unsupported map ${mapType}`);
    const doubles: number[] = [];
    for (let i = 0; i < 18; i++) doubles.push(g.f64());
    const translation: [number, number, number] = [doubles[0]!, doubles[1]!, doubles[2]!];
    const scale = doubles[3]!;

    if (g.u32() !== 1) throw new RuntimeError("OpenVDB: unexpected tree buffer count");
    const background = g.f32();
    const numTiles = g.u32();
    const numChildren = g.u32();
    if (numTiles !== 0) throw new RuntimeError("OpenVDB: root tiles unsupported");

    const leafOrigins: [number, number, number][] = [];
    const leafMasks: Uint8Array[] = [];
    for (let c = 0; c < numChildren; c++) {
        const org5 = g.coord();
        const cm5 = g.raw(4096).slice();
        const vm5 = g.raw(4096).slice();
        readCompressed(g, 32768, vm5, false, background);
        for (let i5 = 0; i5 < 32768; i5++) {
            if (!bit(cm5, i5)) continue;
            const org4: [number, number, number] = [
                org5[0] + ((i5 >> 10) << 7),
                org5[1] + (((i5 >> 5) & 31) << 7),
                org5[2] + ((i5 & 31) << 7),
            ];
            const cm4 = g.raw(512).slice();
            const vm4 = g.raw(512).slice();
            readCompressed(g, 4096, vm4, false, background);
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
        leafValues.push(readCompressed(g, 512, leafMasks[i]!, half, background));
    }
    if (g.o !== found.endPos && g.o !== data.length) throw new RuntimeError(`OpenVDB: buffers ended at ${g.o}`);

    return { translation, scale, background, leafOrigins, leafMasks, leafValues };
}

// ---------------------------------------------------------------------------
// NanoVDB v32.3 builder (see tools/vdb/vdb_to_nvdb.py for the layout notes).
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
        const key = (BigInt((u.origin[2] >> 12) & 0x1fffff)) |
            (BigInt((u.origin[1] >> 12) & 0x1fffff) << 21n) |
            (BigInt((u.origin[0] >> 12) & 0x1fffff) << 42n);
        const to = offRoot + ROOT_DATA_SIZE + i * ROOT_TILE_SIZE;
        view.setBigUint64(to, key, true);
        view.setBigInt64(to + 8, BigInt(upperOff.get(keyOf(u.origin))! - offRoot), true);
        view.setUint32(to + 16, 0, true);
        view.setFloat32(to + 20, 0, true);
    });

    // Internal nodes.
    const writeInternal = (
        base: number, st: SubtreeStats, maskBytes: number, childBits: number[],
        tableOff: number, childOffsets: Map<number, number>,
    ) => {
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
