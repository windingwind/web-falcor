/**
 * PBRTImporter usePBRTMaterials path: the Settings option
 * `PBRTImporter:usePBRTMaterials` (same key as native) maps pbrt material
 * types to the dedicated PBRT material classes instead of StandardMaterial.
 * Verifies the scene instantiates all six types (scene defines), that the
 * full PathTracer megakernel compiles the four newly-reachable material
 * classes (Dielectric/CoatedConductor/CoatedDiffuse/DiffuseTransmission) to
 * WGSL, renders non-black spheres, and that the image differs from the
 * StandardMaterial mapping (different reflectance models) while the area
 * light stays Standard in both.
 */

import {
    runPbrtScene,
    getGlobalSettings,
    RenderGraph,
    createPass,
    type Scene,
    type Device,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const W = 256;
const H = 256;

// Shared scene asset (also rendered by render-native-pbrt-materials.py).
const kSceneUrl = "/tests/oracle/assets/oracle-pbrt-materials.pbrt";

async function render(device: Device, scene: Scene, frames: number): Promise<Float32Array> {
    scene.camera.setAspectRatio(W / H);
    const graph = new RenderGraph(device, "Default");
    graph.onResize(W, H);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1 }), "PathTracer");
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("PathTracer.color", "Accumulate.input");
    graph.markOutput("Accumulate.output");
    graph.setScene(scene);
    for (let f = 0; f < frames; f++) graph.execute(device.renderContext);
    return new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("Accumulate.output")!)).buffer);
}

gpuTest("PbrtMaterials.usePBRTMaterialsPath", async ({ device }) => {
    // Standard mapping first (option off = default).
    const stdScene = await runPbrtScene(device, await (await fetch(kSceneUrl)).text(), "/Falcor/media");
    const stdImg = await render(device, stdScene, 16);

    getGlobalSettings().addOptions({ PBRTImporter: { usePBRTMaterials: true } });
    try {
        const scene = await runPbrtScene(device, await (await fetch(kSceneUrl)).text(), "/Falcor/media");
        const defines = scene.getSceneDefines();
        for (const name of [
            "WEBFALCOR_MTL_PBRT_DIFFUSE",
            "WEBFALCOR_MTL_PBRT_COATED_DIFFUSE",
            "WEBFALCOR_MTL_PBRT_CONDUCTOR",
            "WEBFALCOR_MTL_PBRT_COATED_CONDUCTOR",
            "WEBFALCOR_MTL_PBRT_DIELECTRIC",
            "WEBFALCOR_MTL_PBRT_DIFFUSE_TRANSMISSION",
        ]) {
            expectEq(String(defines.get(name)), "1", `${name} enabled`);
        }
        // Area light + floor stay Standard (native keeps area-light materials Standard).
        expectEq(String(defines.get("WEBFALCOR_MTL_STANDARD")), "1", "Standard still present");

        // 256 accumulated frames: the dielectric/diffusetransmission spheres
        // leave too much variance for block gates at 64 (displaced-cornell lesson).
        const img = await render(device, scene, 256);

        // Radiance oracle: native render with the same graph + option enabled.
        // Regenerate with:
        //   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-pbrt-materials.py --headless
        const res = await fetch("/tests/oracle/out-native/oracle-pbrt-materials.Accumulate.output.0.exr");
        const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        expectEq(width, W, "oracle resolution");
        let oracleGates = { bias: 0, badBlocks: 0 };
        {
            let bias = 0;
            let refSum = 0;
            // 16x16 blocks: at 8x8 the sphere-silhouette/caustic RNG variance
            // still trips the gate at 256 frames (displaced-cornell lesson);
            // the per-region native compare below is the fine-grained assert.
            const block = 16;
            let badBlocks = 0;
            for (let by = 0; by < H / block; by++) {
                for (let bx = 0; bx < W / block; bx++) {
                    let diff = 0;
                    let ref = 0;
                    for (let y = by * block; y < (by + 1) * block; y++) {
                        for (let x = bx * block; x < (bx + 1) * block; x++) {
                            const wi = (y * W + x) * 4;
                            const ni = ((height - 1 - y) * width + x) * 4;
                            for (let c = 0; c < 3; c++) {
                                diff += Math.abs(img[wi + c]! - data[ni + c]!);
                                bias += img[wi + c]! - data[ni + c]!;
                                ref += data[ni + c]!;
                            }
                        }
                    }
                    refSum += ref;
                    if (diff > Math.max(ref, block * block * 3 * 0.02) * 0.1) badBlocks++;
                }
            }
            const n = W * H * 3;
            const meanRef = refSum / n;
            console.error(`# pbrt-materials oracle: bias=${(bias / n).toExponential(2)} meanRef=${meanRef.toFixed(4)} badBlocks=${badBlocks}/256`);
            oracleGates = { bias: bias / n, badBlocks };
        }

        // Per-sphere screen regions (row of spheres at y≈1): sample a horizontal band.
        const bandY0 = Math.floor(H * 0.45);
        const bandY1 = Math.floor(H * 0.62);
        const regionMeanNative = (x0: number, x1: number) => {
            let s = 0;
            let c = 0;
            for (let y = bandY0; y < bandY1; y++) {
                for (let x = x0; x < x1; x++) {
                    const i = ((height - 1 - y) * width + x) * 4;
                    s += data[i]! + data[i + 1]! + data[i + 2]!;
                    c += 3;
                }
            }
            return s / c;
        };
        const regionMean = (img4: Float32Array, x0: number, x1: number) => {
            let s = 0;
            let c = 0;
            for (let y = bandY0; y < bandY1; y++) {
                for (let x = x0; x < x1; x++) {
                    const i = (y * W + x) * 4;
                    s += img4[i]! + img4[i + 1]! + img4[i + 2]!;
                    c += 3;
                }
            }
            return s / c;
        };
        const kRegions: [string, number, number][] = [
            ["diffuse", 0.02, 0.16],
            ["coateddiffuse", 0.2, 0.34],
            ["conductor", 0.37, 0.48],
            ["coatedconductor", 0.52, 0.63],
            ["dielectric", 0.66, 0.78],
            ["diffusetransmission", 0.84, 0.98],
        ];
        let sumRel = 0;
        for (const [name, f0, f1] of kRegions) {
            const m = regionMean(img, Math.floor(W * f0), Math.floor(W * f1));
            const ms = regionMean(stdImg, Math.floor(W * f0), Math.floor(W * f1));
            const mn = regionMeanNative(Math.floor(W * f0), Math.floor(W * f1));
            expectEq(Number.isFinite(m) && m > 1e-3, true, `${name} region lit (mean ${m})`);
            expectEq(Math.abs(m - mn) / Math.max(mn, 1e-2) < 0.05, true, `${name} region matches native (web ${m.toFixed(4)} vs ${mn.toFixed(4)})`);
            sumRel += Math.abs(m - ms) / Math.max(ms, 1e-3);
            console.error(`# pbrt-materials ${name}: pbrt=${m.toFixed(4)} native=${mn.toFixed(4)} std=${ms.toFixed(4)}`);
        }
        // The PBRT material models must actually change the shading vs Standard.
        expectEq(sumRel > 0.05, true, `PBRT vs Standard shading differs (sum rel diff ${sumRel.toFixed(3)})`);
        expectEq(Math.abs(oracleGates.bias) < 3e-3, true, `radiance bias ${oracleGates.bias}`);
        expectEq(oracleGates.badBlocks <= 16, true, `bad 16x16 blocks ${oracleGates.badBlocks}`);
    } finally {
        getGlobalSettings().addOptions({ PBRTImporter: { usePBRTMaterials: false } });
    }
});
