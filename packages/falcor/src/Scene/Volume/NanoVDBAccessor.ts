/**
 * CPU reads of a NanoVDB v32.3 float grid buffer, like nanovdb::ReadAccessor (getValue, probeLeaf), caching the
 * last upper/lower/leaf node visited as NanoVDB's accessor does.
 */

const kTree = 672;
const kRootTiles = 64;
const kRootTileSize = 32;
const kUpperTable = 8256;
const kLowerTable = 1088;
const kLeafValues = 96;

export class NanoVDBAccessor {
    private readonly view: DataView;
    private readonly root: number;
    private readonly background: number;
    private readonly tiles = new Map<string, number>();
    // Cached nodes: origin (x, y, z) and byte offset; -1 when empty.
    private upper = { x: 0, y: 0, z: 0, off: -1 };
    private lower = { x: 0, y: 0, z: 0, off: -1 };
    private leaf = { x: 0, y: 0, z: 0, off: -1 };

    constructor(readonly buffer: Uint8Array) {
        this.view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this.root = kTree + Number(this.view.getBigUint64(kTree + 24, true));
        this.background = this.view.getFloat32(this.root + 28, true);
        const tileCount = this.view.getUint32(this.root + 24, true);
        for (let t = 0; t < tileCount; t++) {
            const o = this.root + kRootTiles + t * kRootTileSize;
            const key = this.view.getBigUint64(o, true);
            const [i, j, k] = [Number((key >> 42n) & 0x1fffffn), Number((key >> 21n) & 0x1fffffn), Number(key & 0x1fffffn)];
            this.tiles.set(`${i},${j},${k}`, o);
        }
    }

    /** Leaf count (tree().nodeCount(0)). */
    get leafCount(): number {
        return this.view.getUint32(kTree + 32, true);
    }

    /** indexBBox(): the root's inclusive index bounds. */
    get indexBBox(): { min: [number, number, number]; max: [number, number, number] } {
        const v = (k: number) => this.view.getInt32(this.root + k * 4, true);
        return { min: [v(0), v(1), v(2)], max: [v(3), v(4), v(5)] };
    }

    getValue(i: number, j: number, k: number): number {
        const leaf = this.probeLeaf(i, j, k);
        if (leaf >= 0) return this.view.getFloat32(leaf + kLeafValues + ((((i & 7) << 6) | ((j & 7) << 3) | (k & 7)) << 2), true);
        return this.tileValue;
    }

    /** Byte offset of the leaf holding (i, j, k), or -1 (then `tileValue` holds the value there). */
    probeLeaf(i: number, j: number, k: number): number {
        const L = this.leaf;
        if (L.off >= 0 && (i & ~7) === L.x && (j & ~7) === L.y && (k & ~7) === L.z) return L.off;
        const lower = this.probeLower(i, j, k);
        if (lower < 0) return -1;
        const n = (((i & 127) >> 3) << 8) | (((j & 127) >> 3) << 4) | ((k & 127) >> 3);
        const entry = lower + kLowerTable + n * 8;
        if (!this.childBit(lower, 512, n)) {
            this.tileValue = this.view.getFloat32(entry, true);
            return -1;
        }
        const off = lower + Number(this.view.getBigInt64(entry, true));
        this.leaf = { x: i & ~7, y: j & ~7, z: k & ~7, off };
        return off;
    }

    /** The value of the tile a probe ended in (the background outside the tree). */
    tileValue = 0;

    /** Leaf values in NanoVDB order (x * 64 + y * 8 + z), as leaf->data()->mValues. */
    leafValues(leaf: number): Float32Array {
        const at = this.buffer.byteOffset + leaf + kLeafValues;
        return at % 4 === 0 ? new Float32Array(this.buffer.buffer, at, 512) : new Float32Array(this.buffer.slice(leaf + kLeafValues, leaf + kLeafValues + 2048).buffer);
    }

    private probeLower(i: number, j: number, k: number): number {
        const L = this.lower;
        if (L.off >= 0 && (i & ~127) === L.x && (j & ~127) === L.y && (k & ~127) === L.z) return L.off;
        const upper = this.probeUpper(i, j, k);
        if (upper < 0) return -1;
        const n = (((i & 4095) >> 7) << 10) | (((j & 4095) >> 7) << 5) | ((k & 4095) >> 7);
        const entry = upper + kUpperTable + n * 8;
        if (!this.childBit(upper, 4096, n)) {
            this.tileValue = this.view.getFloat32(entry, true);
            return -1;
        }
        const off = upper + Number(this.view.getBigInt64(entry, true));
        this.lower = { x: i & ~127, y: j & ~127, z: k & ~127, off };
        return off;
    }

    private probeUpper(i: number, j: number, k: number): number {
        const U = this.upper;
        if (U.off >= 0 && (i & ~4095) === U.x && (j & ~4095) === U.y && (k & ~4095) === U.z) return U.off;
        // Root keys: uint32(ijk) >> 12 per axis.
        const tile = this.tiles.get(`${i >>> 12},${j >>> 12},${k >>> 12}`);
        if (tile === undefined) {
            this.tileValue = this.background;
            return -1;
        }
        const child = Number(this.view.getBigInt64(tile + 8, true));
        if (child === 0) {
            this.tileValue = this.view.getFloat32(tile + 20, true);
            return -1;
        }
        const off = this.root + child;
        this.upper = { x: i & ~4095, y: j & ~4095, z: k & ~4095, off };
        return off;
    }

    /** The child-mask bit `n` of an internal node with `maskBytes`-byte masks (value mask first). */
    private childBit(node: number, maskBytes: number, n: number): boolean {
        return ((this.buffer[node + 32 + maskBytes + (n >> 3)]! >> (n & 7)) & 1) === 1;
    }
}
