#!/usr/bin/env python3
"""OpenVDB .vdb -> NanoVDB .nvdb converter (FloatGrid, NanoVDB v32.3 ABI).

Builds the exact in-memory NanoVDB grid buffer that PNanoVDB.h (bundled with
Falcor, ABI 32.3) traverses, wrapped in the nanovdb::io file container
(codec NONE). Reference for the web TS port; native Mogwai's .nvdb loader
(header-only NanoVDB, unaffected by the broken vcpkg openvdb) validates the
output via Grid python bindings.

Usage: python3 scripts/vdb/vdb_to_nvdb.py in.vdb out.nvdb [--grid density]
"""

import argparse
import struct
import numpy as np

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vdb_parse import parse

ALIGN = 32

GRID_DATA_SIZE = 672
TREE_DATA_SIZE = 64
ROOT_DATA_SIZE = 64      # 48B fields aligned to 32
ROOT_TILE_SIZE = 32
UPPER_SIZE = 8256 + 32768 * 8   # InternalData<5>: header+masks+stats padded, 32^3 tiles
LOWER_SIZE = 1088 + 4096 * 8    # InternalData<4>
LEAF_SIZE = 96 + 512 * 4        # LeafData<3>


def string_hash(s):
    h = 0
    for ch in s.encode():
        overflow = h >> (64 - 8)
        h = (h * 67 + ch + overflow) & 0xFFFFFFFFFFFFFFFF
    return h


def pack_mask(bits):
    return np.packbits(bits.astype(np.uint8), bitorder='little').tobytes()


def build_grid_buffer(grid, gridname):
    leaf_order = grid["leaf_order"]
    masks = grid["masks"]
    values = grid["values"]
    translation = grid["translation"]
    scale = grid["scale"]
    background = grid["background"]

    # --- tree structure (x-major bit order everywhere) -------------------
    uppers = {}   # origin5 -> {bit4 -> origin4}
    lowers = {}   # origin4 -> {bitL -> leaf origin}
    for org in leaf_order:
        o4 = (org[0] & ~127, org[1] & ~127, org[2] & ~127)
        o5 = (org[0] & ~4095, org[1] & ~4095, org[2] & ~4095)
        b4 = (((org[0] & 127) >> 3) << 8) | (((org[1] & 127) >> 3) << 4) | ((org[2] & 127) >> 3)
        b5 = (((o4[0] & 4095) >> 7) << 10) | (((o4[1] & 4095) >> 7) << 5) | ((o4[2] & 4095) >> 7)
        uppers.setdefault(o5, {})[b5] = o4
        lowers.setdefault(o4, {})[b4] = org

    upper_list = sorted(uppers.keys())
    lower_list = sorted({o4 for m in uppers.values() for o4 in m.values()})
    n_upper, n_lower, n_leaf = len(upper_list), len(lower_list), len(leaf_order)

    # Leaves laid out grouped by lower node (breadth-first order).
    leaf_list = []
    for o4 in lower_list:
        for b in sorted(lowers[o4]):
            leaf_list.append(lowers[o4][b])

    # --- per-leaf stats + bboxes ------------------------------------------
    leaf_stats = {}
    total_active = 0
    for org in leaf_list:
        vm = masks[org]
        vals = values[org][vm]
        idx = np.flatnonzero(vm)
        x, y, z = idx >> 6, (idx >> 3) & 7, idx & 7
        bbmin = (org[0] + int(x.min()), org[1] + int(y.min()), org[2] + int(z.min()))
        bbdif = (int(x.max() - x.min()), int(y.max() - y.min()), int(z.max() - z.min()))
        # Sequential f64 summation (not numpy pairwise) so the TS port can
        # reproduce the stats byte-exactly with a plain loop.
        sf = 0.0
        sf2 = 0.0
        for v in vals:
            fv = float(v)
            sf += fv
            sf2 += fv * fv
        cnt = int(vm.sum())
        avg = sf / cnt
        leaf_stats[org] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "avg": avg, "std": max(0.0, sf2 / cnt - avg * avg) ** 0.5,
            "bbmin": bbmin, "bbdif": bbdif, "count": cnt,
            "sum": sf, "sum2": sf2,
        }
        total_active += int(vm.sum())

    def agg(children):
        cnt = sum(c["count"] for c in children)
        s = sum(c["sum"] for c in children)
        s2 = sum(c["sum2"] for c in children)
        avg = s / cnt
        var = max(0.0, s2 / cnt - avg * avg)
        mins = min(c["min"] for c in children)
        maxs = max(c["max"] for c in children)
        bbmin = tuple(min(c["bbmin"][k] for c in children) for k in range(3))
        bbmax = tuple(max(c["bbmin"][k] + (c["bbdif"][k] if "bbdif" in c else c["bbmax"][k] - c["bbmin"][k]) for c in children) for k in range(3))
        return {"min": mins, "max": maxs, "avg": avg, "std": var ** 0.5,
                "bbmin": bbmin, "bbmax": bbmax, "count": cnt, "sum": s, "sum2": s2}

    lower_stats = {o4: agg([leaf_stats[org] for org in lowers[o4].values()]) for o4 in lower_list}
    upper_stats = {o5: agg([lower_stats[o4] for o4 in uppers[o5].values()]) for o5 in upper_list}
    root_stats = agg([upper_stats[o5] for o5 in upper_list])

    # --- offsets (from grid start): grid|tree|root+tiles|upper|lower|leaf --
    off_tree = GRID_DATA_SIZE
    off_root = off_tree + TREE_DATA_SIZE
    off_upper0 = off_root + ROOT_DATA_SIZE + n_upper * ROOT_TILE_SIZE
    off_lower0 = off_upper0 + n_upper * UPPER_SIZE
    off_leaf0 = off_lower0 + n_lower * LOWER_SIZE
    grid_size = off_leaf0 + n_leaf * LEAF_SIZE

    upper_off = {o: off_upper0 + i * UPPER_SIZE for i, o in enumerate(upper_list)}
    lower_off = {o: off_lower0 + i * LOWER_SIZE for i, o in enumerate(lower_list)}
    leaf_off = {o: off_leaf0 + i * LEAF_SIZE for i, o in enumerate(leaf_list)}

    buf = bytearray(grid_size)

    # --- GridData ----------------------------------------------------------
    version = (32 << 21) | (3 << 10) | 3
    # GridFlags: HasBBox(2)|HasMinMax(4)|HasAverage(8)|HasStdDeviation(16)|IsBreadthFirst(32); bit 0 = HasLongGridName!
    flags = 2 | 4 | 8 | 16 | 32
    struct.pack_into('<QQIIIIQ', buf, 0,
                     0x304244566f6e614e, 0xFFFFFFFFFFFFFFFF, version, flags, 0, 1, grid_size)
    name_b = gridname.encode()
    buf[40:40 + len(name_b)] = name_b
    o = 40 + 256
    matf = [scale, 0, 0, 0, scale, 0, 0, 0, scale]
    invf = [1.0 / scale, 0, 0, 0, 1.0 / scale, 0, 0, 0, 1.0 / scale]
    struct.pack_into('<9f', buf, o, *matf); o += 36
    struct.pack_into('<9f', buf, o, *invf); o += 36
    struct.pack_into('<3f', buf, o, *translation); o += 12
    struct.pack_into('<f', buf, o, 1.0); o += 4
    struct.pack_into('<9d', buf, o, *matf); o += 72
    struct.pack_into('<9d', buf, o, *invf); o += 72
    struct.pack_into('<3d', buf, o, *translation); o += 24
    struct.pack_into('<d', buf, o, 1.0); o += 8
    assert o == 40 + 256 + 264
    ib_min, ib_max = root_stats["bbmin"], root_stats["bbmax"]
    world_min = [ib_min[k] * scale + translation[k] for k in range(3)]
    world_max = [(ib_max[k] + 1) * scale + translation[k] for k in range(3)]
    struct.pack_into('<6d', buf, o, *world_min, *world_max); o += 48
    struct.pack_into('<3d', buf, o, scale, scale, scale); o += 24
    struct.pack_into('<II', buf, o, 2, 1); o += 8  # GridClass::FogVolume, GridType::Float
    struct.pack_into('<qI', buf, o, 0, 0)  # no blind metadata

    # --- TreeData (offsets relative to TREE start) --------------------------
    tile_counts = [0, 0, 0]  # active value-tiles per level (leaf-parent=lower, etc.)
    struct.pack_into('<4Q3I3IQ', buf, off_tree,
                     off_leaf0 - off_tree, off_lower0 - off_tree, off_upper0 - off_tree, off_root - off_tree,
                     n_leaf, n_lower, n_upper,
                     *tile_counts, total_active)

    # --- RootData + tiles ----------------------------------------------------
    struct.pack_into('<6iIfffff', buf, off_root,
                     *ib_min, *ib_max, n_upper,
                     background, root_stats["min"], root_stats["max"], root_stats["avg"], root_stats["std"])
    for i, o5 in enumerate(upper_list):
        key = ((o5[2] >> 12) & 0x1FFFFF) | (((o5[1] >> 12) & 0x1FFFFF) << 21) | (((o5[0] >> 12) & 0x1FFFFF) << 42)
        to = off_root + ROOT_DATA_SIZE + i * ROOT_TILE_SIZE
        struct.pack_into('<QqIf', buf, to, key, upper_off[o5] - off_root, 0, 0.0)

    # --- Upper nodes (InternalData<5>) --------------------------------------
    for o5 in upper_list:
        base = upper_off[o5]
        st = upper_stats[o5]
        struct.pack_into('<6iQ', buf, base, *st["bbmin"], *st["bbmax"], 0)
        cmask = np.zeros(32768, bool)
        for b in uppers[o5]:
            cmask[b] = True
        buf[base + 32:base + 32 + 4096] = b'\0' * 4096          # valueMask: no active tiles
        buf[base + 32 + 4096:base + 32 + 8192] = pack_mask(cmask)
        struct.pack_into('<ffff', buf, base + 8224, st["min"], st["max"], st["avg"], st["std"])
        table = base + 8256
        for b, o4 in uppers[o5].items():
            struct.pack_into('<q', buf, table + b * 8, lower_off[o4] - base)

    # --- Lower nodes (InternalData<4>) --------------------------------------
    for o4 in lower_list:
        base = lower_off[o4]
        st = lower_stats[o4]
        struct.pack_into('<6iQ', buf, base, *st["bbmin"], *st["bbmax"], 0)
        cmask = np.zeros(4096, bool)
        for b in lowers[o4]:
            cmask[b] = True
        buf[base + 32:base + 32 + 512] = b'\0' * 512
        buf[base + 32 + 512:base + 32 + 1024] = pack_mask(cmask)
        struct.pack_into('<ffff', buf, base + 1056, st["min"], st["max"], st["avg"], st["std"])
        table = base + 1088
        for b, org in lowers[o4].items():
            struct.pack_into('<q', buf, table + b * 8, leaf_off[org] - base)

    # --- Leaves --------------------------------------------------------------
    for org in leaf_list:
        base = leaf_off[org]
        st = leaf_stats[org]
        struct.pack_into('<3i3BB', buf, base, *st["bbmin"], *st["bbdif"], 0)
        buf[base + 16:base + 80] = pack_mask(masks[org])
        struct.pack_into('<ffff', buf, base + 80, st["min"], st["max"], st["avg"], st["std"])
        buf[base + 96:base + 96 + 2048] = values[org].astype('<f4').tobytes()

    meta = {
        "gridSize": grid_size, "voxelCount": total_active,
        "indexBBox": (ib_min, ib_max), "worldBBox": (world_min, world_max),
        "nodeCount": (n_leaf, n_lower, n_upper, 1), "voxelSize": scale,
    }
    return bytes(buf), meta


def write_nvdb(path, grid_buf, meta, gridname):
    version = (32 << 21) | (3 << 10) | 3
    with open(path, 'wb') as f:
        f.write(struct.pack('<QIHH', 0x304244566f6e614e, version, 1, 0))  # Header, codec NONE
        name_b = gridname.encode() + b'\0'
        f.write(struct.pack('<4Q', meta["gridSize"], meta["gridSize"], string_hash(gridname), meta["voxelCount"]))
        f.write(struct.pack('<II', 1, 2))  # GridType::Float, GridClass::FogVolume
        (wmin, wmax) = meta["worldBBox"]
        f.write(struct.pack('<6d', *wmin, *wmax))
        (imin, imax) = meta["indexBBox"]
        f.write(struct.pack('<6i', *imin, *imax))
        f.write(struct.pack('<3d', meta["voxelSize"], meta["voxelSize"], meta["voxelSize"]))
        f.write(struct.pack('<I', len(name_b)))
        f.write(struct.pack('<4I', *meta["nodeCount"]))
        f.write(struct.pack('<3I', 0, 0, 0))
        f.write(struct.pack('<HHI', 0, 0, version))  # codec, padding, version
        f.write(name_b)
        f.write(grid_buf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("outfile")
    ap.add_argument("--grid", default="density")
    args = ap.parse_args()
    grid = parse(args.infile, args.grid)
    buf, meta = build_grid_buffer(grid, args.grid)
    write_nvdb(args.outfile, buf, meta, args.grid)
    print(f"wrote {args.outfile}: gridSize={meta['gridSize']} voxels={meta['voxelCount']} "
          f"nodes={meta['nodeCount']} indexBBox={meta['indexBBox']}")


if __name__ == "__main__":
    main()
