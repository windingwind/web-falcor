/**
 * OVERALL VERIFY — upstream image test: the UNMODIFIED PathTracerDielectrics.py
 * graph over its own upstream scene (nested_dielectrics.pyscene: nested glass/
 * water volumes via nestedPriority + volumeAbsorption, rotated env map,
 * maxSurfaceBounces=20, stratified VBufferRT jitter), 4 accumulated frames,
 * diffed against the native PNG capture.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-imgtest-dielectrics.py --headless
 *
 * (Historical: this test exposed the SceneBuilder multi-instancing bug —
 * one mesh instanced N times rendered only its last instance.)
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("ImageTestDielectrics.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracerDielectrics.py")).text();
    const [graph] = await runGraphScript(device, source);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/nested_dielectrics.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 4; f++) graph!.execute(ctx);

    // ToneMapper.dst is RGBA8UnormSrgb: readback bytes compare directly to PNG.
    const web = new Uint8Array((await ctx.readTextureSubresource(graph!.getOutput("ToneMapper.dst")!)).buffer);
    const blob = await (await fetch("/tests/oracle/out-native/oracle-imgtest-dielectrics.ToneMapper.dst.0.png")).blob();
    const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
    expectEq(bitmap.width, size, "oracle resolution");
    const canvas = new OffscreenCanvas(size, size);
    const c2d = canvas.getContext("2d", { willReadFrequently: true })!;
    c2d.drawImage(bitmap, 0, 0);
    const nat = c2d.getImageData(0, 0, size, size).data;

    let mse = 0;
    let bad = 0;
    const bandMse = [0, 0, 0, 0];
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = y * size + x;
            let pixelMax = 0;
            for (let ch = 0; ch < 3; ch++) {
                const d = (web[i * 4 + ch]! - nat[i * 4 + ch]!) / 255;
                mse += d * d;
                bandMse[Math.floor(y / 64)]! += d * d;
                pixelMax = Math.max(pixelMax, Math.abs(d));
            }
            if (pixelMax > 0.05) bad++;
        }
    }
    mse /= size * size * 3;
    console.error(`# imgtestDielectrics: mse=${mse.toExponential(2)} bad=${bad} bands=${bandMse.map((v) => (v / (size * 64 * 3)).toExponential(1)).join(",")}`);
    for (const [px, py] of [[128, 16], [40, 40], [128, 128], [128, 230]]) {
        const i = (py! * size + px!) * 4;
        console.error(`#   px(${px},${py}) web=${web[i]},${web[i + 1]},${web[i + 2]} nat=${nat[i]},${nat[i + 1]},${nat[i + 2]}`);
    }
    // Stochastic content (20-bounce glass paths): cornell bad-pixel policy.
    expectEq(mse < 5e-4, true, `sRGB MSE ${mse}`);
    expectEq(bad <= 400, true, `bad pixels ${bad}`);
});
