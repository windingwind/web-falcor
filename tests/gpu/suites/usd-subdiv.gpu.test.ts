/**
 * USD subdivision surfaces: meshes with a refinement level (the prim's refinementLevel,
 * the stage's rtx:hydra:refinementLevel, or a Settings override) are refined through
 * OpenSubdiv's Bfr (compiled to wasm) exactly like native's tessellate(): limit
 * positions, normals from the partials, face-varying/vertex texcoords. Catmull-Clark,
 * Loop and bilinear schemes, compared with a native GBufferRT capture.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-usd-subdiv.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("UsdSubdiv.matchesNativeOracle", async ({ device }) => {
    const [w, h] = [256, 128];
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/usd-subdiv.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    expectEq(scene.stats.instances, 5, "five meshes");

    const graph = new RenderGraph(device, "UsdSubdiv");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
    for (const c of ["mask", "posW", "normW", "faceNormalW", "texC"]) graph.markOutput(`GBufferRT.${c}`);
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const native = async (c: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/usd-subdiv.GBufferRT.${c}.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
    const read = async (c: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${c}`)!)).buffer);
    const [natMask, webMask] = [await native("mask"), await read("mask")];
    let hits = 0;
    let maskMismatch = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            // Native EXR captures are bottom-up.
            const nat = natMask[((h - 1 - y) * w + x) * 4] !== 0;
            if (nat) hits++;
            if (nat !== (webMask[y * w + x] !== 0)) maskMismatch++;
        }
    }
    console.error(`# usdSubdiv: hits ${hits}, mask mismatches ${maskMismatch}`);
    expectEq(hits > 500, true, `the scene is hit (${hits})`);
    expectEq(maskMismatch <= 8, true, `hit masks match (${maskMismatch})`);
    // The capture is half precision.
    for (const [c, tol] of [["posW", 1e-2], ["normW", 2e-3], ["faceNormalW", 2e-3], ["texC", 2e-3]] as const) {
        const [nat, web] = [await native(c), await read(c)];
        let max = 0;
        let bad = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const ni = ((h - 1 - y) * w + x) * 4;
                const wi = (y * w + x) * (web.length / (w * h));
                if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                let d = 0;
                for (let k = 0; k < (c === "texC" ? 2 : 3); k++) d = Math.max(d, Math.abs(web[wi + k]! - nat[ni + k]!));
                // Silhouette pixels may hit different surfaces.
                if (d > tol) bad++;
                max = Math.max(max, d);
            }
        }
        console.error(`# usdSubdiv.${c}: max=${max.toExponential(2)} bad=${bad}`);
        expectEq(bad <= 8, true, `${c}: ${bad} pixels beyond ${tol}`);
    }
});
