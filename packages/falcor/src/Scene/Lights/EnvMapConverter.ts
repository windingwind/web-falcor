/**
 * Equal-area octahedral -> lat-long environment map conversion, mirroring
 * plugins/importers/PBRTImporter/EnvMapConverter (pbrt-v4 stores environment
 * images in Clarberg's equal-area octahedral mapping; Falcor's EnvMap is
 * lat-long).
 *
 * Web divergence (docs §9): native runs EnvMapConverter.cs.slang, whose
 * partial `.rgb` store into an RGBA32Float RWTexture2D WGSL cannot express.
 * The port decodes env maps on the CPU anyway, so the same per-pixel mapping
 * runs there, with the kernel's linear/clamp sampler reproduced as bilinear
 * filtering with clamp-to-edge addressing.
 */

import { RuntimeError } from "../../Core/Error.js";
import type { HdrImage } from "../../Utils/Image/HDRDecoder.js";

/** Mirrors MathHelpers.slang latlong_map_to_world. */
export function latlongMapToWorld(u: number, v: number): [number, number, number] {
    const saturate = (x: number) => Math.min(Math.max(x, 0), 1);
    const phi = Math.PI * (2 * saturate(u) - 1);
    const theta = Math.PI * saturate(v);
    const sinTheta = Math.sin(theta);
    return [sinTheta * Math.sin(phi), Math.cos(theta), -sinTheta * Math.cos(phi)];
}

/** Mirrors MathHelpers.slang ndir_to_oct_equal_area_unorm. */
export function ndirToOctEqualAreaUnorm(n: [number, number, number]): [number, number] {
    const r = Math.sqrt(1 - Math.abs(n[2]));
    const phi = Math.atan2(Math.abs(n[1]), Math.abs(n[0]));
    let py = r * phi * (2 / Math.PI);
    let px = r - py;
    // Reflect over the diagonals for the lower hemisphere.
    if (n[2] < 0) [px, py] = [1 - py, 1 - px];
    // signNonZero: zero counts as positive.
    px *= n[0] >= 0 ? 1 : -1;
    py *= n[1] >= 0 ? 1 : -1;
    return [px * 0.5 + 0.5, py * 0.5 + 0.5];
}

/** Bilinear lookup with clamp-to-edge addressing, as a linear/clamp SampleLevel. */
function sampleBilinearClamp(image: HdrImage, u: number, v: number, out: Float32Array, at: number): void {
    const x = u * image.width - 0.5;
    const y = v * image.height - 0.5;
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = x - x0;
    const fy = y - y0;
    const clampX = (i: number) => Math.min(Math.max(i, 0), image.width - 1);
    const clampY = (i: number) => Math.min(Math.max(i, 0), image.height - 1);
    const texel = (ix: number, iy: number) => (clampY(iy) * image.width + clampX(ix)) * 4;
    const t00 = texel(x0, y0);
    const t10 = texel(x0 + 1, y0);
    const t01 = texel(x0, y0 + 1);
    const t11 = texel(x0 + 1, y0 + 1);
    for (let c = 0; c < 3; c++) {
        const top = image.data[t00 + c]! * (1 - fx) + image.data[t10 + c]! * fx;
        const bottom = image.data[t01 + c]! * (1 - fx) + image.data[t11 + c]! * fx;
        out[at + c] = top * (1 - fy) + bottom * fy;
    }
}

/**
 * Mirrors EnvMapConverter::convertEqualAreaOctToLatLong: a square octahedral
 * map of size N becomes a 2N x N lat-long map. Alpha stays 0, as the kernel
 * writes only rgb into a zero-initialised texture.
 */
export function convertEqualAreaOctToLatLong(src: HdrImage): HdrImage {
    if (src.width !== src.height) throw new RuntimeError(`EnvMapConverter: octahedral maps are square (got ${src.width}x${src.height})`);
    const width = src.width * 2;
    const height = src.height;
    const data = new Float32Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dir = latlongMapToWorld((x + 0.5) / width, (y + 0.5) / height);
            const [u, v] = ndirToOctEqualAreaUnorm(dir);
            sampleBilinearClamp(src, u, v, data, (y * width + x) * 4);
        }
    }
    return { width, height, data };
}
