#!/usr/bin/env python3
"""OpenVDB .vdb parser (FloatGrid, 5-4-3 tree) — reference implementation.

Scope: exactly what Falcor's volume assets need (validated on
media/test_scenes/volumes/smoke.vdb): file version 222, ZIP absent,
COMPRESS_ACTIVE_MASK, half-float leaf values, UniformScaleTranslateMap.
This is the reference for the web TS port (packages/falcor OpenVDB loader);
it also generates ground-truth samples and the .nvdb conversion input.

Background: the native prebuilt openvdb (vcpkg, Debug AND Release) segfaults
in openvdb::initialize() on this machine (ODR/static split-brain between
libFalcor's inlined template instantiations and libopenvdb.so.9.0), so
native ground truth via Grid.createFromOpenVDBFile is impossible here.
Validation instead: byte-exact stream consumption (topology ends exactly at
the descriptor's blockPos, buffers exactly at end-of-file) + the file's own
metadata (file_voxel_count, file_bbox_min/max) + NanoVDB round-trip through
native Mogwai's .nvdb loader.

Usage:
  python3 scripts/vdb/vdb_parse.py <file.vdb> [--samples out.json] [--seed 42]
"""

import argparse
import json
import struct
import numpy as np


class Reader:
    def __init__(self, data, o=0):
        self.d = data
        self.o = o

    def u32(self):
        v = struct.unpack_from('<I', self.d, self.o)[0]
        self.o += 4
        return v

    def i32(self):
        v = struct.unpack_from('<i', self.d, self.o)[0]
        self.o += 4
        return v

    def i64(self):
        v = struct.unpack_from('<q', self.d, self.o)[0]
        self.o += 8
        return v

    def f32(self):
        v = struct.unpack_from('<f', self.d, self.o)[0]
        self.o += 4
        return v

    def f64(self):
        v = struct.unpack_from('<d', self.d, self.o)[0]
        self.o += 8
        return v

    def byte(self):
        v = self.d[self.o]
        self.o += 1
        return v

    def raw(self, n):
        v = self.d[self.o:self.o + n]
        self.o += n
        return v

    def s(self):
        return self.raw(self.u32()).decode('ascii', 'replace')

    def coord(self):
        return (self.i32(), self.i32(), self.i32())


def load_mask(r, nbits):
    return np.unpackbits(np.frombuffer(r.raw(nbits // 8), np.uint8), bitorder='little').astype(bool)


def read_compressed(r, count, valuemask, half):
    """io::readCompressedValues with COMPRESS_ACTIVE_MASK (no ZIP/BLOSC)."""
    meta = r.byte()
    assert meta < 7, f"bad compression metadata byte {meta}"
    rd = (lambda: float(np.frombuffer(r.raw(2), '<f2')[0])) if half else (lambda: r.f32())
    inactive0 = None
    if meta in (2, 4):
        inactive0 = rd()
    elif meta == 5:
        inactive0 = rd()
        rd()  # second inactive value (unused: background reconstruction)
    selmask = None
    if meta in (3, 4, 5):
        selmask = load_mask(r, count)
    out = np.zeros(count, np.float32)
    if meta == 6:  # NO_MASK_AND_ALL_VALS
        out[:] = np.frombuffer(r.raw(count * (2 if half else 4)), '<f2' if half else '<f4').astype(np.float32)
        return out
    readmask = valuemask.copy()
    if selmask is not None:
        readmask &= ~selmask
    n = int(readmask.sum())
    out[readmask] = np.frombuffer(r.raw(n * (2 if half else 4)), '<f2' if half else '<f4').astype(np.float32)
    if selmask is not None and inactive0 is not None:
        out[selmask & valuemask] = inactive0
    return out


def parse(path, gridname="density"):
    data = open(path, 'rb').read()
    r = Reader(data)
    assert r.i64() == 0x56444220, "not an OpenVDB file"
    file_version = r.u32()
    assert file_version >= 222, f"unsupported OpenVDB file version {file_version}"
    r.u32(), r.u32()  # library version
    has_grid_offsets = r.byte()
    assert has_grid_offsets
    r.raw(36)  # uuid
    for _ in range(r.u32()):  # file metadata
        r.s(), r.s(), r.raw(r.u32())

    grids = {}
    for _ in range(r.u32()):
        name = r.s().split('\x1e')[0]
        gtype = r.s()
        r.s()  # instance parent
        grid_pos, block_pos, end_pos = r.i64(), r.i64(), r.i64()
        grids[name] = (gtype, grid_pos, block_pos, end_pos)
    assert gridname in grids, f"grid '{gridname}' not in {list(grids)}"
    gtype, grid_pos, block_pos, end_pos = grids[gridname]
    half = gtype.endswith("_HalfFloat")
    base = gtype[:-len("_HalfFloat")] if half else gtype
    assert base == "Tree_float_5_4_3", f"unsupported grid type {gtype}"

    g = Reader(data, grid_pos)
    compression = g.u32()
    assert compression == 2, f"only COMPRESS_ACTIVE_MASK supported, got flags {compression}"
    meta = {}
    for _ in range(g.u32()):
        mname, mtype = g.s(), g.s()
        meta[mname] = (mtype, g.raw(g.u32()))
    map_type = g.s()
    assert map_type == "UniformScaleTranslateMap", map_type
    doubles = [g.f64() for _ in range(18)]
    translation, scale = doubles[0:3], doubles[3]

    assert g.u32() == 1  # tree buffer count
    background = g.f32()
    num_tiles, num_children = g.u32(), g.u32()
    assert num_tiles == 0, "root tiles unsupported"

    leaves = {}
    leaf_order = []
    for _ in range(num_children):
        org5 = g.coord()
        cm5 = load_mask(g, 32768)
        vm5 = load_mask(g, 32768)
        read_compressed(g, 32768, vm5, False)  # internal tile values (unused: fog)
        for i5 in np.flatnonzero(cm5):
            n5 = int(i5)
            x, y, z = n5 >> 10, (n5 >> 5) & 31, n5 & 31
            org4 = (org5[0] + (x << 7), org5[1] + (y << 7), org5[2] + (z << 7))
            cm4 = load_mask(g, 4096)
            vm4 = load_mask(g, 4096)
            read_compressed(g, 4096, vm4, False)
            for i4 in np.flatnonzero(cm4):
                n4 = int(i4)
                xx, yy, zz = n4 >> 8, (n4 >> 4) & 15, n4 & 15
                org_leaf = (org4[0] + (xx << 3), org4[1] + (yy << 3), org4[2] + (zz << 3))
                leaves[org_leaf] = load_mask(g, 512)
                leaf_order.append(org_leaf)
    assert g.o == block_pos, f"topology ended at {g.o}, expected {block_pos}"

    values = {}
    for org in leaf_order:
        vm = load_mask(g, 512)
        assert (vm == leaves[org]).all()
        values[org] = read_compressed(g, 512, vm, half)
    assert g.o == end_pos or g.o == len(data), f"buffers ended at {g.o}"

    return {
        "translation": translation,
        "scale": scale,
        "background": background,
        "leaf_order": leaf_order,
        "masks": leaves,
        "values": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--grid", default="density")
    ap.add_argument("--samples", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    grid = parse(args.file, args.grid)
    total = sum(int(m.sum()) for m in grid["masks"].values())
    mins = [min(o[k] for o in grid["leaf_order"]) for k in range(3)]
    maxs = [max(o[k] + 7 for o in grid["leaf_order"]) for k in range(3)]
    print(f"leaves={len(grid['leaf_order'])} activeVoxels={total} leafBBox={mins}..{maxs}")
    print(f"translation={grid['translation']} scale={grid['scale']}")

    if args.samples:
        rng = np.random.default_rng(args.seed)
        samples = []
        orgs = grid["leaf_order"]
        for _ in range(500):
            org = orgs[int(rng.integers(len(orgs)))]
            idx = int(rng.integers(512))
            x, y, z = idx >> 6, (idx >> 3) & 7, idx & 7
            samples.append([org[0] + x, org[1] + y, org[2] + z,
                            float(grid["values"][org][idx]), bool(grid["masks"][org][idx])])
        json.dump({
            "file": args.file.split('/')[-1],
            "grid": args.grid,
            "activeVoxels": total,
            "leafCount": len(orgs),
            "translation": grid["translation"],
            "scale": grid["scale"],
            "samples": samples,
        }, open(args.samples, 'w'))
        print(f"wrote {args.samples}")


if __name__ == "__main__":
    main()
