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
/** Script, scene, captured file, max relative L1 of the (tone-mapped or debug) output, max mean ratio error. */
const kBistro = "Bistro_v5_2/BistroExterior.pyscene";
const kOtherScenes = ["Bistro_v5_2/BistroInterior_Wine.pyscene", "EmeraldSquare_v4_1/EmeraldSquare_Day.pyscene", "SunTemple_v4/SunTemple/SunTemple.pyscene", "ZeroDay_v1/ZeroDay.pyscene"];
const kScripts: [string, string, string, number, number][] = [
    ["PathTracer", kBistro, "PathTracer.ToneMapper.dst.0.png", 0.1, 0.02],
    ["MinimalPathTracer", kBistro, "MinimalPathTracer.ToneMapper.dst.0.png", 0.1, 0.02],
    // The stock RTXDI graph disables accumulation: one noisy ReSTIR frame.
    ["RTXDI", kBistro, "RTXDI.ToneMapper.dst.0.png", 0.3, 0.03],
    ["SceneDebugger", kBistro, "SceneDebugger.SceneDebugger.output.0.exr", 0.02, 0.01],
    ...kOtherScenes.flatMap((scene): [string, string, string, number, number][] => [
        // ZeroDay is nearly black after tone mapping: its 8-bit output (values 0-2) is dominated by quantized noise;
        // big-scenes-pt compares its linear radiance instead.
        ["PathTracer", scene, "PathTracer.ToneMapper.dst.0.png", 0.2, scene.startsWith("ZeroDay") ? 0.1 : 0.03],
        ["SceneDebugger", scene, "SceneDebugger.SceneDebugger.output.0.exr", 0.02, 0.01],
    ]),
];

async function load(bytes: Uint8Array, name: string): Promise<{ data: ArrayLike<number>; width: number; height: number; stride: number }> {
    if (name.endsWith(".png")) {
        const img = await decodePng(bytes);
        return { data: img.data, width: img.width, height: img.height, stride: img.channels };
    }
    const exr = parseExr(bytes.slice().buffer, 1015) as { data: Float32Array; width: number; height: number };
    return { data: exr.data, width: exr.width, height: exr.height, stride: 4 };
}

for (const [script, scene, file, maxRel, maxRatio] of kScripts) {
    const sceneName = scene.slice(scene.lastIndexOf("/") + 1, -".pyscene".length);
    const dir = scene === kBistro ? script : `${script}-${sceneName}`;
    gpuTest(`BigScenes.scriptMatchesNative.${dir}`, async ({ device }) => {
        if (!kBigScenes) throw new SkipError("run with EXTRA_QUERY=bigScenes=1");
        await initScripting("/node_modules/pyodide");
        const template = await (await fetch("/tests/oracle/render-native-big-script.py")).text();
        const source = template.replace(/^kScript = .*$/m, `kScript = "${script}"`).replace(/^kScene = .*$/m, `kScene = "${scene}"`);
        const { frameCapture, scene: loaded } = await runMogwaiSource(device, source, "/tests/oracle", { fileName: "render-native-big-script.py" });
        loaded?.destroy(); // production scenes are large: free each before the next test loads its own
        const captured = frameCapture.captured.find((f) => f.name === file);
        expectEq(captured !== undefined, true, `captured ${file} (got ${frameCapture.captured.map((f) => f.name).join(", ")})`);
        const web = await load(captured!.bytes, file);
        const nat = await load(new Uint8Array(await (await fetch(`/tests/oracle/out-native/big-scripts/${dir}/${file}`)).arrayBuffer()), file);
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
        console.error(`# big-script ${dir}: relL1 ${rel.toFixed(4)}, mean web/native ${ratio.map((r) => r.toFixed(3)).join(",")}`);
        for (const r of ratio) expectEq(Math.abs(r - 1) <= maxRatio, true, `mean ratio ${ratio}`);
        expectEq(rel <= maxRel, true, `relative L1 ${rel}`);
    });
}
