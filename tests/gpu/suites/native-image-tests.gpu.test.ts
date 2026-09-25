/**
 * Unmodified native image-test scripts (Falcor/tests/image_tests) compared against native
 * Mogwai's own captures of the same scripts (tests/oracle/out-native/image-tests, captured like
 * run_image_tests.py: outputDir + m.script from the test's directory, --headless --precise).
 * Native compares with ImageCompare mse at tolerance 0 against its own references; across
 * implementations each file gets an mse bound. The raster-G-buffer scripts can't be captured
 * natively here (GBufferRaster needs ROVs this Vulkan device lacks).
 */

import { compareImages, initScripting, loadCompareImage } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { runMogwaiScript, runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import { gpuTest, expectEq } from "../harness/registry.js";

interface Case {
    /** Per-file mse bound. */
    tolerance: number;
    /** Web frames standing in for native ones (native frame -> web frame), with the script's frame list rewritten. */
    frames?: { from: string; to: string; map: Record<number, number> };
}

const kCases: Record<string, Case> = {
    // Native's skinned pose trails its clock by one frame after the first (web frame N-1 matches
    // native frame N to 1.5e-6); the web skins at the current time.
    "renderpasses/Skinning": { tolerance: 1e-5, frames: { from: "frames=[1,16,64]", to: "frames=[1,15,63]", map: { 1: 1, 16: 15, 64: 63 } } },
    "scene/CameraAnimation": { tolerance: 1e-4 },
    // Path-traced at 1 spp per frame: sampling noise differs, the means agree to ~1e-4.
    "scene/USDPreviewSurface": { tolerance: 5e-3 },
    "scene/Volumes": { tolerance: 1e-6 },
};

for (const [script, c] of Object.entries(kCases)) {
    gpuTest(`NativeImageCompare.${script.replace("/", "_")}`, async ({ device }) => {
        await initScripting("/node_modules/pyodide");
        const [dir, name] = script.split("/");
        const url = `/Falcor/tests/image_tests/${dir}/test_${name}.py`;
        const run = c.frames
            ? await runMogwaiSource(device, (await (await fetch(url)).text()).replace(c.frames.from, c.frames.to), url.slice(0, url.lastIndexOf("/")))
            : await runMogwaiScript(device, url);
        const captured = run.frameCapture.captured;
        expectEq(captured.length > 0, true, "captured images");
        const webName = (nativeName: string) => (c.frames ? nativeName.replace(/\.(\d+)\.(\w+)$/, (_m, f: string, ext: string) => `.${c.frames!.map[Number(f)]}.${ext}`) : nativeName);
        const report: string[] = [];
        let worst = 0;
        const nativeNames = c.frames ? captured.map((f) => f.name.replace(/\.(\d+)\.(\w+)$/, (_m, f: string, ext: string) => `.${Object.entries(c.frames!.map).find(([, w]) => w === Number(f))![0]}.${ext}`)) : captured.map((f) => f.name);
        for (const nativeName of nativeNames) {
            const f = captured.find((x) => x.name === webName(nativeName))!;
            const res = await fetch(`/tests/oracle/out-native/image-tests/${script}/${nativeName}`);
            expectEq(res.ok, true, `native reference for ${nativeName}`);
            const [web, native] = await Promise.all([loadCompareImage(f.bytes, f.name), loadCompareImage(new Uint8Array(await res.arrayBuffer()), nativeName)]);
            const { error } = compareImages(native, web, "mse");
            worst = Math.max(worst, error);
            report.push(`${nativeName}: mse ${error.toExponential(2)}`);
        }
        console.error(`# ${script}: ${report.join("; ")}`);
        expectEq(worst <= c.tolerance, true, `every capture within mse ${c.tolerance} (worst ${worst.toExponential(2)})`);
    });
}
