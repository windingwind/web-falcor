/**
 * Scene.replace_material from a Testbed script: Falcor's scripts/python/test_replace_material.py
 * minus its NeuralMaterial replacement (not part of Falcor 8.0), i.e.
 * tests/oracle/render-native-replace-material.py run unmodified. Cornell box material 0 becomes a
 * PBRTDiffuseMaterial with a checker base-color texture loaded at runtime; the path-traced
 * image after 64 accumulated frames is compared with native's (8x8-block means).
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Testbed.replaceMaterialMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const result = await runTestbedScript(device, "/tests/oracle/render-native-replace-material.py", { argv: ["replace_material.exr"], maxFrames: 1000 });
    const testbed = result.testbeds[0]!;
    const graph = testbed.renderGraph!;
    const out = graph.getOutput(graph.getOutputNames()[0]!)!;
    const [w, h] = [out.width, out.height];
    const web = new Float32Array((await device.renderContext.readTextureSubresource(out)).buffer);
    const nat = (parseExr(await (await fetch("/tests/oracle/out-native/replace-material.AccumulatePass.output.64.exr")).arrayBuffer(), 1015) as { data: Float32Array }).data;
    expectEq([w, h].join("x"), "256x256", "output size");
    const block = 8;
    let abs = 0, ref = 0, checker = 0;
    for (let by = 0; by < h / block; by++)
        for (let bx = 0; bx < w / block; bx++)
            for (let c = 0; c < 3; c++) {
                let a = 0, b = 0;
                for (let y = by * block; y < (by + 1) * block; y++)
                    for (let x = bx * block; x < (bx + 1) * block; x++) {
                        a += web[(y * w + x) * 4 + c]!;
                        b += nat[((h - 1 - y) * w + x) * 4 + c]!; // EXR rows are bottom-up
                    }
                abs += Math.abs(a - b);
                ref += Math.abs(b);
            }
    // The replaced material is textured: its checker shows up as local contrast within blocks.
    for (let i = 0; i < w * h - 1; i++) checker = Math.max(checker, Math.abs(web[i * 4]! - web[(i + 1) * 4]!));
    const rel = abs / ref;
    console.error(`# replace-material: 8x8-block relL1 ${rel.toExponential(2)}, max neighbour step ${checker.toFixed(2)}`);
    expectEq(rel < 0.05, true, `block relative L1 ${rel}`);
});
