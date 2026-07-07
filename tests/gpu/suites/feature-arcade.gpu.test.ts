/**
 * FEATURE VERIFY — FBX import (Arcade) vs native. The upstream GBufferRT.py
 * image-test graph over the upstream Arcade.pyscene (its real test scene):
 * validates the web FBX importer (assimpjs + ported AssimpImporter mapping)
 * end-to-end — node transforms, geometry, material colors and textures —
 * through GBuffer channels.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-feature-arcade.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("FeatureArcade.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/GBufferRT.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/Arcade/Arcade.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/Arcade");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const compare = async (ref: string, oracle: string, comps: number, meanTol: number, badTol: number, badThreshold = 1e-3) => {
        const tex = graph!.getOutput(ref);
        if (!tex) {
            console.error(`# ${oracle}: output missing`);
            expectEq(true, false, `${ref} missing`);
            return;
        }
        const web = new Float32Array((await ctx.readTextureSubresource(tex)).buffer);
        const res = await fetch(`/tests/oracle/out-native/${oracle}`);
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        const webComps = web.length / (size * size);
        let sum = 0;
        let bad = 0;
        let maxD = 0;
        let maxAt = [0, 0];
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * webComps;
                const ni = ((height - 1 - y) * width + x) * 4;
                let pixelMax = 0;
                for (let c = 0; c < comps; c++) {
                    const w = Math.min(Math.max(web[wi + c]!, -65504), 65504);
                    const n = Math.min(Math.max(data[ni + c]!, -65504), 65504);
                    const d = Math.abs(w - n);
                    sum += d;
                    pixelMax = Math.max(pixelMax, d);
                }
                if (pixelMax > badThreshold) bad++;
                if (pixelMax > maxD) {
                    maxD = pixelMax;
                    maxAt = [x, y];
                }
            }
        }
        const mean = sum / (size * size * comps);
        const mi = (maxAt[1]! * size + maxAt[0]!) * webComps;
        const mn = ((height - 1 - maxAt[1]!) * width + maxAt[0]!) * 4;
        console.error(`# ${oracle}: mean=${mean.toExponential(2)} bad=${bad} max=${maxD.toExponential(2)} at=(${maxAt}) web=${web[mi]?.toFixed(3)},${web[mi + 1]?.toFixed(3)},${web[mi + 2]?.toFixed(3)} nat=${data[mn]?.toFixed(3)},${data[mn + 1]?.toFixed(3)},${data[mn + 2]?.toFixed(3)}`);
        if (bad > badTol) {
            // Coarse 16x16 bad-pixel density map (row 0 = image top).
            const map: number[][] = Array.from({ length: 16 }, () => Array.from({ length: 16 }, () => 0));
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const wi = (y * size + x) * webComps;
                    const ni = ((height - 1 - y) * width + x) * 4;
                    let pm = 0;
                    for (let c = 0; c < comps; c++) pm = Math.max(pm, Math.abs(web[wi + c]! - data[ni + c]!));
                    if (pm > badThreshold) map[y >> 4]![x >> 4]!++;
                }
            }
            for (let r = 0; r < 16; r++) console.error(`# map ${r.toString().padStart(2)}: ${map[r]!.map((v) => (v > 128 ? "#" : v > 32 ? "+" : v > 0 ? "." : " ")).join("")}`);
        }
        expectEq(mean < meanTol, true, `${oracle} mean ${mean}`);
        expectEq(bad <= badTol, true, `${oracle} bad ${bad}`);
    };

    // Geometry first (posW/faceNormalW), then materials (diffuse), then texturing.
    await compare("GBufferRT.posW", "oracle-feature-arcade.GBufferRT.posW.0.exr", 3, 1e-3, 120, 1e-2);
    await compare("GBufferRT.faceNormalW", "oracle-feature-arcade.GBufferRT.faceNormalW.0.exr", 3, 1e-3, 120, 1e-2);
    await compare("GBufferRT.texC", "oracle-feature-arcade.GBufferRT.texC.0.exr", 2, 1e-3, 120, 1e-2);
    // Residual bad pixels are atlas-layer wrap seams: textures smaller than
    // the packed array emulate repeat addressing via frac(), which cannot
    // bilinear-blend across the seam like native hardware wrap (documented).
    await compare("GBufferRT.diffuseOpacity", "oracle-feature-arcade.GBufferRT.diffuseOpacity.0.exr", 3, 2e-3, 400, 1e-2);
    // Emissive is scaled x150 (getMaterial edit): sub-texel ray precision at
    // the high-contrast screen border flips bilinear neighborhoods, so the
    // bad-pixel gate uses an absolute threshold of 1.0 (~0.7% of peak).
    await compare("GBufferRT.emissive", "oracle-feature-arcade.GBufferRT.emissive.0.exr", 3, 2e-2, 400, 1.0);
});
