/**
 * USD composition through the USD importer: the layer is composed (sublayers, external
 * and internal references, instanceable prims, variant selections) before the render
 * scene is built, as a USD stage does natively. Compared with a native GBufferRT
 * capture; the referenced part's own material binding is checked through the diffuse
 * albedo, and unbound meshes through native's display-color default material.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-usd-compose.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("UsdCompose.matchesNativeOracle", async ({ device }) => {
    const [w, h] = [192, 96];
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/usd-compose.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    // The sublayer's floor, the referenced quad, the selected variant and two instanced triangles.
    expectEq(scene.stats.instances, 5, "every composed mesh is imported");
    expectEq(scene.getMaterial(0)?.name, "Red", "the referenced material is bound");
    const floor = scene.materials.find((m) => m.name?.startsWith("default-mesh") && m.basic.baseColor?.y === 0.6);
    expectEq(floor?.basic.baseColor?.x, 0.2, "unbound meshes take their display color (the floor's)");

    const graph = new RenderGraph(device, "UsdCompose");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
    for (const c of ["mask", "posW", "normW", "diffuseOpacity"]) graph.markOutput(`GBufferRT.${c}`);
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const native = async (c: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/usd-compose.GBufferRT.${c}.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
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
    console.error(`# usdCompose: hits ${hits}, mask mismatches ${maskMismatch}`);
    expectEq(hits > 500, true, `the scene is hit (${hits})`);
    expectEq(maskMismatch <= 8, true, `hit masks match (${maskMismatch})`);
    // The capture is half precision.
    for (const [c, tol] of [["posW", 1e-2], ["normW", 2e-3], ["diffuseOpacity", 2e-3]] as const) {
        const [nat, web] = [await native(c), await read(c)];
        let max = 0;
        let bad = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const ni = ((h - 1 - y) * w + x) * 4;
                const wi = (y * w + x) * 4;
                if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                let d = 0;
                for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(web[wi + k]! - nat[ni + k]!));
                // Silhouette pixels may hit different surfaces.
                if (d > tol) bad++;
                max = Math.max(max, d);
            }
        }
        console.error(`# usdCompose.${c}: max=${max.toExponential(2)} bad=${bad}`);
        expectEq(bad <= 8, true, `${c}: ${bad} pixels beyond ${tol}`);
    }
});
