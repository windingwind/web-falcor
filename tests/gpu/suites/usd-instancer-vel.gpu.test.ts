/**
 * USD PointInstancers with velocities, accelerations and angularVelocities, compared with
 * native GBufferRT captures at 0.5s, 1.25s and 1.83s. Native samples the instance transforms at
 * the positions' sample times with baseTime = EarliestTime, so aligned velocities extrapolate
 * from the first sample. Default-valued or offset velocities are ignored, and native warns
 * about them.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-usd-instancer-vel.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("UsdInstancerVel.matchesNativeOracle", async ({ device }) => {
    const [w, h] = [192, 96];
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/usd-instancer-vel.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    expectEq(scene.hasAnimation(), true, "the time samples animate the scene");

    const graph = new RenderGraph(device, "UsdInstancerVel");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
    for (const c of ["mask", "posW", "normW"]) graph.markOutput(`GBufferRT.${c}`);
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    const read = async (c: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${c}`)!)).buffer);
    const native = async (c: string, frame: number) => (parseExr(await (await fetch(`/tests/oracle/out-native/usd-instancer-vel.GBufferRT.${c}.${frame}.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;

    for (const frame of [12, 30, 44]) {
        scene.animate(frame / 24);
        graph.execute(ctx);
        const [natMask, webMask] = [await native("mask", frame), await read("mask")];
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
        const errors: string[] = [];
        for (const [c, tol] of [["posW", 1e-2], ["normW", 2e-3]] as const) {
            const [nat, web] = [await native(c, frame), await read(c)];
            let bad = 0;
            const region = [0, 0, 0];
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const ni = ((h - 1 - y) * w + x) * 4;
                    const wi = (y * w + x) * 4;
                    if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                    let d = 0;
                    for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(web[wi + k]! - nat[ni + k]!));
                    // Silhouette pixels may hit different surfaces.
                    if (d > tol) { bad++; const X = nat[ni]!; region[X < -1 ? 0 : X < 2 ? 1 : 2]!++; }
                }
            }
            errors.push(`${c} bad=${bad} [sampled ${region[0]}, default ${region[1]}, offset ${region[2]}]`);
            expectEq(bad <= 8, true, `frame ${frame}: ${c} ${bad} pixels beyond ${tol}`);
        }
        console.error(`# usdInstancerVel frame ${frame}: hits ${hits}, mask mismatches ${maskMismatch}, ${errors.join(", ")}`);
        expectEq(hits > 500, true, `frame ${frame}: the scene is hit (${hits})`);
        expectEq(maskMismatch <= 8, true, `frame ${frame}: hit masks match (${maskMismatch})`);
    }
});
