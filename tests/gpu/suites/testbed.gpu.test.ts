/**
 * Falcor's Python Testbed examples (Falcor/scripts/python), run unmodified: the script's
 * `while not testbed.should_close: testbed.frame()` loop runs headless for a few frames.
 */

import { initScripting, runTestbedScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Testbed.ballsSimulationScript", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const { testbeds } = await runTestbedScript(device, "/Falcor/scripts/python/balls/balls.py", { maxFrames: 5 });
    expectEq(testbeds.length, 1, "the script created one Testbed");
    const testbed = testbeds[0]!;
    expectEq(testbed.getFrameCount(), 5, "frames until should_close");
    const tex = testbed.renderTexture!;
    const bytes = await device.renderContext.readTextureSubresource(tex, 0);
    const px = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    let lit = 0;
    for (let i = 0; i < px.length; i += 4) if (px[i]! + px[i + 1]! + px[i + 2]! > 0.05) lit++;
    // 100 balls of radius 0.1 in [-1, 1]^2 cover roughly pi * 0.01 / 4 of the image each (overlaps aside).
    console.error(`# testbed balls: ${tex.width}x${tex.height}, lit pixels ${lit} (${((lit / (tex.width * tex.height)) * 100).toFixed(1)}%)`);
    expectEq(lit > tex.width * tex.height * 0.2 && lit < tex.width * tex.height * 0.9, true, "balls drawn over part of the image");
});
