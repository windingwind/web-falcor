/**
 * Falcor's stock graph scripts (Falcor/scripts/*.py) run unchanged on BistroExterior, 16 frames at t = 0, against
 * native captures of the same generated script. Opt-in: EXTRA_QUERY=bigScenes=1.
 *
 * Regenerate the oracle per script (see the header of tests/oracle/render-native-big-script.py).
 */

import { initScripting } from "@web-falcor/falcor";
import { decodePng } from "../../../packages/falcor/src/Utils/Image/PNGCodec.js";
import parseExr from "parse-exr";
import { runMogwaiSource } from "../../../packages/mogwai/src/ScriptRunner.js";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const kBigScenes = new URLSearchParams(location.search).get("bigScenes") === "1";
/** Script, captured file, max relative L1 of the (tone-mapped or debug) output, max mean ratio error. */
const kScripts: [string, string, number, number][] = [
    ["PathTracer", "PathTracer.ToneMapper.dst.0.png", 0.1, 0.02],
    ["MinimalPathTracer", "MinimalPathTracer.ToneMapper.dst.0.png", 0.1, 0.02],
    // The stock RTXDI graph disables accumulation: one noisy ReSTIR frame.
    ["RTXDI", "RTXDI.ToneMapper.dst.0.png", 0.3, 0.03],
    ["SceneDebugger", "SceneDebugger.SceneDebugger.output.0.exr", 0.02, 0.01],
];

async function load(bytes: Uint8Array, name: string): Promise<{ data: ArrayLike<number>; width: number; height: number; stride: number }> {
    if (name.endsWith(".png")) {
        const img = await decodePng(bytes);
        return { data: img.data, width: img.width, height: img.height, stride: img.channels };
    }
    const exr = parseExr(bytes.slice().buffer, 1015) as { data: Float32Array; width: number; height: number };
    return { data: exr.data, width: exr.width, height: exr.height, stride: 4 };
}

for (const [script, file, maxRel, maxRatio] of kScripts) {
    gpuTest(`BigScenes.scriptMatchesNative.${script}`, async ({ device }) => {
        if (!kBigScenes) throw new SkipError("run with EXTRA_QUERY=bigScenes=1");
        await initScripting("/node_modules/pyodide");
        const template = await (await fetch("/tests/oracle/render-native-big-script.py")).text();
        const source = template.replace(/^kScript = .*$/m, `kScript = "${script}"`);
        const { frameCapture } = await runMogwaiSource(device, source, "/tests/oracle", { fileName: "render-native-big-script.py" });
        const captured = frameCapture.captured.find((f) => f.name === file);
        expectEq(captured !== undefined, true, `captured ${file} (got ${frameCapture.captured.map((f) => f.name).join(", ")})`);
        const web = await load(captured!.bytes, file);
        const nat = await load(new Uint8Array(await (await fetch(`/tests/oracle/out-native/big-scripts/${script}/${file}`)).arrayBuffer()), file);
        expectEq(`${web.width}x${web.height}`, `${nat.width}x${nat.height}`, "size");
        // EXR rows are stored bottom-up by parse-exr; both files come from the same writer, so compare as stored.
        let sum = 0, refSum = 0;
        const mean = [0, 0, 0], refMean = [0, 0, 0];
        for (let i = 0; i < web.width * web.height; i++) {
            for (let c = 0; c < 3; c++) {
                const [a, b] = [Number(web.data[i * web.stride + c]), Number(nat.data[i * nat.stride + c])];
                if (!Number.isFinite(a) || !Number.isFinite(b)) continue;
                sum += Math.abs(a - b);
                refSum += Math.abs(b);
                mean[c] += a;
                refMean[c] += b;
            }
        }
        const rel = sum / Math.max(refSum, 1e-9);
        const ratio = mean.map((m, c) => m / Math.max(refMean[c]!, 1e-9));
        console.error(`# big-script ${script}: relL1 ${rel.toFixed(4)}, mean web/native ${ratio.map((r) => r.toFixed(3)).join(",")}`);
        for (const r of ratio) expectEq(Math.abs(r - 1) <= maxRatio, true, `mean ratio ${ratio}`);
        expectEq(rel <= maxRel, true, `relative L1 ${rel}`);
    });
}
