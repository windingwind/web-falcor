#!/usr/bin/env python3
"""Generates the Bitmap decoder fixtures in packages/falcor/tests/fixtures/bitmap/.

Each image comes with the pixels it must decode to (<name>.expected.json),
written from the same numpy arrays, so the TS decoders are checked against an
independent writer: Pillow for the common cases and a hand-rolled PNG writer
for what Pillow can't produce (16-bit RGB, Adam7 interlace, 2-bit gray).

Usage: python3 scripts/gen-bitmap-fixtures.py
"""
import json
import os
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

OUT = Path(os.environ.get("BITMAP_FIXTURES", Path(__file__).resolve().parent.parent / "packages/falcor/tests/fixtures/bitmap"))
W, H = 7, 5  # odd sizes exercise row padding and partial Adam7 passes
rng = np.random.default_rng(1234)


def save_expected(name, fmt, pixels):
    (OUT / f"{name}.expected.json").write_text(json.dumps({"width": W, "height": H, "format": fmt, "data": [int(v) for v in pixels.reshape(-1)]}))


def png_chunk(kind, body):
    return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)


def raw_png(path, samples, depth, color_type, interlace=False):
    """samples: (H, W, C) integer array; rows are written with filter 0 (or Adam7 passes)."""
    def pack_rows(img):
        rows = b""
        for row in img:
            if depth == 16:
                data = b"".join(struct.pack(">H", int(v)) for v in row.reshape(-1))
            elif depth == 8:
                data = bytes(int(v) for v in row.reshape(-1))
            else:
                bits = "".join(format(int(v), f"0{depth}b") for v in row.reshape(-1))
                bits += "0" * (-len(bits) % 8)
                data = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
            rows += b"\x00" + data
        return rows

    if interlace:
        raw = b""
        for x0, y0, dx, dy in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]:
            sub = samples[y0::dy, x0::dx]
            if sub.size:
                raw += pack_rows(sub)
    else:
        raw = pack_rows(samples)
    ihdr = struct.pack(">IIBBBBB", W, H, depth, color_type, 0, 0, 1 if interlace else 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", ihdr) + png_chunk(b"IDAT", zlib.compress(raw, 9)) + png_chunk(b"IEND", b""))


def bgra(rgb, alpha=None):
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8) if alpha is None else alpha[..., None]
    return np.concatenate([rgb[..., 2:3], rgb[..., 1:2], rgb[..., 0:1], a], axis=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (H, W), dtype=np.uint8)
    gray = rng.integers(0, 256, (H, W), dtype=np.uint8)

    # Pillow-written 8-bit cases.
    Image.fromarray(gray).save(OUT / "gray8.png")
    save_expected("gray8", "R8Unorm", gray)
    Image.fromarray(rgb).save(OUT / "rgb8.png")
    save_expected("rgb8", "BGRX8Unorm", bgra(rgb))
    rgba = np.concatenate([rgb, alpha[..., None]], axis=2)
    Image.fromarray(rgba).save(OUT / "rgba8.png")
    save_expected("rgba8", "BGRA8Unorm", bgra(rgb, alpha))
    Image.fromarray(np.stack([gray, alpha], axis=2)).save(OUT / "gray-alpha8.png")
    save_expected("gray-alpha8", "BGRA8Unorm", bgra(np.stack([gray] * 3, axis=2), alpha))

    # Palette with a tRNS table: FreeImage expands it to 32-bit BGRA.
    pal = rng.integers(0, 256, (4, 3), dtype=np.uint8)
    idx = rng.integers(0, 4, (H, W), dtype=np.uint8)
    img = Image.frombytes("P", (W, H), idx.tobytes())
    img.putpalette(pal.reshape(-1).tolist())
    trns = [255, 0, 128, 64]
    img.save(OUT / "palette.png", transparency=bytes(trns))
    save_expected("palette", "BGRA8Unorm", bgra(pal[idx], np.array(trns, np.uint8)[idx]))

    # A palette that is a gray ramp reads as gray (FIC_MINISBLACK -> R8Unorm).
    img = Image.frombytes("P", (W, H), gray.tobytes())
    img.putpalette([v for i in range(256) for v in (i, i, i)])
    img.save(OUT / "palette-gray-ramp.png")
    save_expected("palette-gray-ramp", "R8Unorm", gray)

    # BMP: 24-bit, 8-bit gray ramp, 8-bit colour palette (browser-decoded; GPU suite).
    Image.fromarray(rgb).save(OUT / "rgb24.bmp")
    save_expected("rgb24", "BGRX8Unorm", bgra(rgb))
    Image.fromarray(gray).save(OUT / "gray8.bmp")
    save_expected("gray8-bmp", "R8Unorm", gray)
    img = Image.frombytes("P", (W, H), idx.tobytes())
    img.putpalette(pal.reshape(-1).tolist() + [0] * (768 - pal.size))
    img.save(OUT / "palette8.bmp")
    save_expected("palette8", "BGRA8Unorm", bgra(pal[idx]))

    # JPEG at quality 100; expected pixels are Pillow's decode (decoders may differ by a step or two).
    smooth = np.clip(np.add.outer(np.arange(H) * 20, np.arange(W) * 12)[..., None] + np.array([0, 40, 80]), 0, 255).astype(np.uint8)
    Image.fromarray(smooth).save(OUT / "rgb.jpg", quality=100, subsampling=0)
    save_expected("rgb-jpg", "BGRX8Unorm", bgra(np.array(Image.open(OUT / "rgb.jpg"))))
    Image.fromarray(smooth[..., 0]).save(OUT / "gray.jpg", quality=100)
    save_expected("gray-jpg", "R8Unorm", np.array(Image.open(OUT / "gray.jpg")))

    # Hand-rolled: 16-bit gray and RGB, Adam7, 2-bit gray.
    gray16 = rng.integers(0, 65536, (H, W), dtype=np.uint16)
    raw_png(OUT / "gray16.png", gray16[..., None], 16, 0)
    save_expected("gray16", "R16Unorm", gray16)
    rgb16 = rng.integers(0, 65536, (H, W, 3), dtype=np.uint16)
    raw_png(OUT / "rgb16.png", rgb16, 16, 2)
    save_expected("rgb16", "RGBA16Unorm", np.concatenate([rgb16, np.full((H, W, 1), 65535, np.uint16)], axis=2))
    raw_png(OUT / "rgba8-adam7.png", rgba, 8, 6, interlace=True)
    save_expected("rgba8-adam7", "BGRA8Unorm", bgra(rgb, alpha))
    gray2 = rng.integers(0, 4, (H, W), dtype=np.uint8)
    raw_png(OUT / "gray2.png", gray2[..., None], 2, 0)
    save_expected("gray2", "R8Unorm", gray2 * 85)

    # TGA (Pillow writes bottom-left origin, uncompressed unless asked).
    Image.fromarray(rgb).save(OUT / "rgb.tga")
    save_expected("rgb", "BGRX8Unorm", bgra(rgb))
    Image.fromarray(rgba).save(OUT / "rgba-rle.tga", compression="tga_rle")
    save_expected("rgba-rle", "BGRA8Unorm", bgra(rgb, alpha))

    # PFM, little endian, rows bottom to top.
    f = rng.standard_normal((H, W, 3)).astype(np.float32) * 10
    (OUT / "float.pfm").write_bytes(f"PF\n{W} {H}\n-1.0\n".encode() + f[::-1].astype("<f4").tobytes())
    expected = np.concatenate([f, np.ones((H, W, 1), np.float32)], axis=2)
    (OUT / "float.expected.json").write_text(json.dumps({"width": W, "height": H, "format": "RGBA32Float", "data": [float(v) for v in expected.reshape(-1)]}))


if __name__ == "__main__":
    main()
