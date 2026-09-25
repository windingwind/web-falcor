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
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

interface Case {
    /** Per-file mse bound. */
    tolerance: number;
    /** Alternatively, the largest fraction of pixels off by more than 1e-3 (edge ties, HDR outliers). */
    outliers?: number;
    /** Stochastic renders: compare 8x8 block means (the noise realizations differ) with this mse bound. */
    block?: number;
    /** Native's capture is unusable on this machine. */
    nativeBroken?: string;
    /** Web frames standing in for native ones (native frame -> web frame), with the script's frame list rewritten. */
    frames?: { from: string; to: string; map: Record<number, number> };
}

/** Scripts with native captures (committed ones, or captured locally with scripts/capture-native-image-tests.sh). */
const kScripts = (
    "renderpasses/BSDFViewer renderpasses/ColorMapPass renderpasses/CompositePass renderpasses/CrossFadePass renderpasses/FLIPPass renderpasses/GaussianBlur renderpasses/GBufferRT " +
    "renderpasses/GBufferRTInline renderpasses/GBufferRTTexGrads renderpasses/MinimalPathTracer renderpasses/ModulateIllumination renderpasses/MVecRT renderpasses/PathTracer " +
    "renderpasses/PathTracerAdaptive renderpasses/PathTracerDielectrics renderpasses/PathTracerMaterials renderpasses/PathTracerReload renderpasses/RTXDI renderpasses/SideBySide " +
    "renderpasses/SimplePostFX renderpasses/Skinning renderpasses/SplitScreen renderpasses/TextureLOD renderpasses/ToneMapping renderpasses/VBufferRT renderpasses/VBufferRTInline " +
    "renderpasses/WARDiffPathTracerMaterialFwd renderpasses/WARDiffPathTracerTranslationBwd renderpasses/WARDiffPathTracerTranslationFwd renderscripts/BSDFViewer " +
    "renderscripts/MinimalPathTracer renderscripts/PathTracer renderscripts/RTXDI renderscripts/SceneDebugger renderscripts/WARDiffPathTracer scene/AnimationBehavior " +
    "scene/CameraAnimation scene/Displacement scene/NDSDFGrids scene/RtProgram scene/SceneCache scene/SDFSBS scene/USDPreviewSurface scene/Volumes"
).split(" ");

const kStochastic = { tolerance: 1e-4, block: 1e-4 };
const kSdfBroken = "native SDF grids render no hits on this machine (a black box); FeatureNDSDFGrid/FeatureSBSGrid check the algorithm instead";
const kCases: Record<string, Case> = {
    // Native's skinned pose trails its clock by one frame after the first (web frame N-1 matches
    // native frame N to 1.5e-6); the web skins at the current time.
    "renderpasses/Skinning": { tolerance: 1e-5, outliers: 0, frames: { from: "frames=[1,16,64]", to: "frames=[1,15,63]", map: { 1: 1, 16: 15, 64: 63 } } },
    "scene/Volumes": { tolerance: 1e-6, outliers: 0 },
    // Stochastic: the noise realizations differ where RNG streams do, the block means must agree.
    "scene/USDPreviewSurface": { tolerance: 1e-4, block: 5e-4 },
    // Ray-cone texture filtering and displaced-surface hits differ at scattered pixels.
    "renderpasses/TextureLOD": { tolerance: 1e-4, block: 1e-4 },
    "scene/Displacement": { tolerance: 1e-4, block: 1e-4 },
    "renderpasses/MinimalPathTracer": kStochastic,
    "renderpasses/PathTracer": kStochastic,
    "renderpasses/PathTracerAdaptive": kStochastic,
    "renderpasses/PathTracerDielectrics": kStochastic,
    "renderpasses/PathTracerMaterials": kStochastic,
    "renderpasses/PathTracerReload": kStochastic,
    // RTXDI without accumulation: one frame of spatiotemporal resampling, very noisy.
    "renderpasses/RTXDI": { tolerance: 1e-4, block: 5e-4 },
    "renderpasses/WARDiffPathTracerTranslationBwd": kStochastic,
    "renderpasses/WARDiffPathTracerTranslationFwd": kStochastic,
    "renderscripts/MinimalPathTracer": kStochastic,
    "renderscripts/PathTracer": kStochastic,
    "renderscripts/RTXDI": { tolerance: 1e-4, block: 5e-4 },
    "renderscripts/WARDiffPathTracer": kStochastic,
    // The grey room's camera is animated: 1 spp per frame at frame 64.
    "scene/SceneCache": { tolerance: 1e-4, block: 2e-4 },
    "scene/NDSDFGrids": { tolerance: 0, nativeBroken: kSdfBroken },
    "scene/SDFSBS": { tolerance: 0, nativeBroken: kSdfBroken },
};

/** mse of the 8x8 block means (RGB): noise cancels, systematic differences remain. */
function blockMse(a: { width: number; height: number; data: Float32Array }, b: { data: Float32Array }): number {
    const [bw, bh] = [Math.floor(a.width / 8), Math.floor(a.height / 8)];
    let sum = 0;
    for (let by = 0; by < bh; by++)
        for (let bx = 0; bx < bw; bx++)
            for (let c = 0; c < 3; c++) {
                let d = 0;
                for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) {
                    const i = ((by * 8 + y) * a.width + bx * 8 + x) * 4 + c;
                    d += a.data[i]! - b.data[i]!;
                }
                sum += (d / 64) ** 2;
            }
    return sum / (bw * bh * 3);
}

/** The committed capture sets run in every suite; the rest (~20 min) with --filter NativeImageCompare. */
const kCommitted = new Set(["renderpasses/Skinning", "scene/CameraAnimation", "scene/USDPreviewSurface", "scene/Volumes"]);
const kSelected = new URLSearchParams(location.search).get("filter")?.includes("NativeImageCompare") ?? false;

for (const script of kScripts) {
    const c: Case = kCases[script] ?? { tolerance: 1e-4 };
    gpuTest(`NativeImageCompare.${script.replace("/", "_")}`, async ({ device }) => {
        if (c.nativeBroken) throw new SkipError(c.nativeBroken);
        if (!kCommitted.has(script) && !kSelected) throw new SkipError("run with --filter NativeImageCompare");
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
        const missing: string[] = [];
        let worst = 0;
        let worstOutliers = 0;
        const failed: string[] = [];
        const nativeNames = c.frames ? captured.map((f) => f.name.replace(/\.(\d+)\.(\w+)$/, (_m, f: string, ext: string) => `.${Object.entries(c.frames!.map).find(([, w]) => w === Number(f))![0]}.${ext}`)) : captured.map((f) => f.name);
        for (const nativeName of nativeNames) {
            const f = captured.find((x) => x.name === webName(nativeName))!;
            const res = await fetch(`/tests/oracle/out-native/image-tests/${script}/${nativeName}`);
            if (!res.ok) {
                missing.push(nativeName);
                continue;
            }
            const [web, native] = await Promise.all([loadCompareImage(f.bytes, f.name), loadCompareImage(new Uint8Array(await res.arrayBuffer()), nativeName)]);
            // HDR (EXR) values compare after x / (1 + |x|), so a few bright outliers don't dominate.
            if (nativeName.endsWith(".exr")) for (const im of [web, native]) im.data.forEach((v, i) => (im.data[i] = v / (1 + Math.abs(v))));
            const { error, errorMap } = compareImages(native, web, "mse");
            // Fraction of pixels off by more than 1e-3 (edge ties and HDR outliers dominate mse across implementations).
            const outliers = errorMap.filter((e) => e > 1e-3).length / errorMap.length;
            const blockErr = blockMse(native, web);
            const ok = error <= c.tolerance || outliers <= (c.outliers ?? 5e-3) || (c.block !== undefined && blockErr <= c.block);
            if (!ok) failed.push(nativeName);
            worst = Math.max(worst, error);
            worstOutliers = Math.max(worstOutliers, outliers);
            report.push(`${nativeName}: mse ${error.toExponential(2)} out ${outliers.toExponential(1)} block ${blockErr.toExponential(1)}`);
        }
        if (report.length === 0) throw new SkipError(`no native captures (scripts/capture-native-image-tests.sh ${script})`);
        console.error(`# ${script}: ${report.join("; ")}${missing.length ? `; not captured natively: ${missing.join(", ")}` : ""}`);
        expectEq(failed.join(", "), "", `captures beyond mse ${c.tolerance}, outliers ${c.outliers ?? 5e-3}${c.block !== undefined ? ` and block mse ${c.block}` : ""} (worst mse ${worst.toExponential(2)}, outliers ${worstOutliers.toExponential(1)})`);
    });
}
