/**
 * HiZBuild: every mip of the min/max pyramid equals a CPU reduction of mip 0 (bit for bit), mip 0 equals the depth
 * buffer, and the chain has the expected sizes, for power-of-two and odd frame sizes.
 */

import { hiZMipCount } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { log, makeScene } from "./stylize-helpers.js";
import { buildGraph, coverage } from "./fx-helpers.js";

/** CPU reduction with the shader's footprint rule (odd sources: the last texel also covers the left-over row/column). */
function reduce(src: Float32Array, sw: number, sh: number): { data: Float32Array; w: number; h: number } {
    const w = Math.max(1, sw >> 1), h = Math.max(1, sh >> 1);
    const data = new Float32Array(w * h * 2);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const x1 = Math.min(2 * x + (x === w - 1 && sw % 2 === 1 ? 2 : 1), sw - 1), y1 = Math.min(2 * y + (y === h - 1 && sh % 2 === 1 ? 2 : 1), sh - 1);
            let lo = Infinity, hi = -Infinity;
            for (let yy = Math.min(2 * y, sh - 1); yy <= y1; yy++) {
                for (let xx = Math.min(2 * x, sw - 1); xx <= x1; xx++) {
                    lo = Math.min(lo, src[(yy * sw + xx) * 2]!);
                    hi = Math.max(hi, src[(yy * sw + xx) * 2 + 1]!);
                }
            }
            data[(y * w + x) * 2] = lo;
            data[(y * w + x) * 2 + 1] = hi;
        }
    }
    return { data, w, h };
}

gpuTest("HiZBuild.pyramidMatchesCpu", async ({ device }) => {
    const scene = makeScene(device);
    for (const [W, H] of [[128, 128], [100, 75], [67, 129]] as const) {
        const g = buildGraph(device, scene, [["HiZBuild", "Z", {}]], [["G.depth", "Z.depth"]], ["Z.hiz", "G.depth"], W, H);
        for (let i = 0; i < 5; i++) {
            g.execute(device.renderContext);
            if ((await coverage(device, g)) > 0.2) break;
        }
        const hiz = g.getOutput("Z.hiz")!;
        const readMip = async (m: number) => {
            const raw = await device.renderContext.readTextureSubresource(hiz, m);
            return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice();
        };
        expectEq(hiz.mipCount, hiZMipCount(W, H), `mip count ${W}x${H}`);
        const m0 = await readMip(0);
        // mip 0 = depth buffer (r = g = depth).
        const dr = await device.renderContext.readTextureSubresource(g.getOutput("G.depth")!);
        const depth = new Float32Array(dr.buffer, dr.byteOffset, dr.byteLength / 4);
        let m0Bad = 0, bg = 0;
        for (let i = 0; i < W * H; i++) {
            if (m0[i * 2] !== depth[i] || m0[i * 2 + 1] !== depth[i]) m0Bad++;
            if (depth[i] === 1) bg++;
        }
        let prev = { data: m0, w: W, h: H }, bad = 0, sizes = `${W}x${H}`;
        for (let m = 1; m < hiz.mipCount; m++) {
            const ref = reduce(prev.data, prev.w, prev.h);
            const gpu = await readMip(m);
            expectEq(gpu.length, ref.data.length, `mip ${m} size`);
            for (let i = 0; i < ref.data.length; i++) if (gpu[i] !== ref.data[i]) bad++;
            prev = { data: gpu, w: ref.w, h: ref.h };
            sizes += ` ${ref.w}x${ref.h}`;
        }
        const top = prev.data;
        let gmin = Infinity, gmax = -Infinity;
        for (let i = 0; i < W * H; i++) { gmin = Math.min(gmin, depth[i]!); gmax = Math.max(gmax, depth[i]!); }
        log(`HiZBuild ${W}x${H}: mips ${hiz.mipCount} (${sizes}), mip0 mismatches ${m0Bad}, background px ${bg}, reduction mismatches ${bad}, top (${top[0]}, ${top[1]}) vs global (${gmin}, ${gmax})`);
        expectEq(m0Bad, 0, "mip 0 = depth");
        expectEq(bad, 0, "every mip = CPU min/max reduction");
        expectEq(top[0] === gmin && top[1] === gmax, true, "1x1 mip = global min/max (conservative footprints)");
        expectEq(bg > 0 && bg < W * H, true, "frame has geometry and background");
    }
});
