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
import { gpuTest, expectEq } from "../harness/registry.js";

const W = 256;
const H = 256;

// Six spheres in a row (one per material type) over a diffuse floor, lit by
// an emissive quad. Left-handed pbrt coords; camera looks down -z (kInvertZ).
const kScene = `
LookAt 0 1.5 6.5  0 1 0  0 1 0
Camera "perspective" "float fov" [40]
WorldBegin
AttributeBegin
  AreaLightSource "diffuse" "rgb L" [14 14 14]
  Translate 0 4.6 0
  Shape "trianglemesh"
    "point3 P" [-1.5 0 -1.5  1.5 0 -1.5  1.5 0 1.5  -1.5 0 1.5]
    "integer indices" [0 1 2 0 2 3]
AttributeEnd
Material "diffuse" "rgb reflectance" [0.7 0.7 0.7]
Shape "trianglemesh"
  "point3 P" [-8 0 -8  8 0 -8  8 0 8  -8 0 8]
  "integer indices" [0 2 1 0 3 2]
AttributeBegin
  Material "diffuse" "rgb reflectance" [0.8 0.1 0.1]
  Translate -3.4 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
AttributeBegin
  Material "coateddiffuse" "rgb reflectance" [0.1 0.5 0.8] "float roughness" [0.1]
  Translate -2.0 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
AttributeBegin
  Material "conductor" "float roughness" [0.05]
  Translate -0.7 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
AttributeBegin
  Material "coatedconductor" "float conductor.roughness" [0.1] "float interface.roughness" [0.05]
  Translate 0.7 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
AttributeBegin
  Material "dielectric" "float eta" [1.5]
  Translate 2.0 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
AttributeBegin
  Material "diffusetransmission" "rgb reflectance" [0.6 0.5 0.1] "rgb transmittance" [0.2 0.4 0.6]
  Translate 3.4 1 0
  Shape "sphere" "float radius" [0.6]
AttributeEnd
`;

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
    const stdScene = await runPbrtScene(device, kScene, "/Falcor/media");
    const stdImg = await render(device, stdScene, 16);

    getGlobalSettings().addOptions({ PBRTImporter: { usePBRTMaterials: true } });
    try {
        const scene = await runPbrtScene(device, kScene, "/Falcor/media");
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

        const img = await render(device, scene, 16);

        // Per-sphere screen regions (row of spheres at y≈1): sample a horizontal band.
        const bandY0 = Math.floor(H * 0.45);
        const bandY1 = Math.floor(H * 0.62);
        const regionMean = (data: Float32Array, x0: number, x1: number) => {
            let s = 0;
            let c = 0;
            for (let y = bandY0; y < bandY1; y++) {
                for (let x = x0; x < x1; x++) {
                    const i = (y * W + x) * 4;
                    s += data[i]! + data[i + 1]! + data[i + 2]!;
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
            expectEq(Number.isFinite(m) && m > 1e-3, true, `${name} region lit (mean ${m})`);
            sumRel += Math.abs(m - ms) / Math.max(ms, 1e-3);
            console.error(`# pbrt-materials ${name}: pbrt=${m.toFixed(4)} std=${ms.toFixed(4)}`);
        }
        // The PBRT material models must actually change the shading vs Standard.
        expectEq(sumRel > 0.05, true, `PBRT vs Standard shading differs (sum rel diff ${sumRel.toFixed(3)})`);
    } finally {
        getGlobalSettings().addOptions({ PBRTImporter: { usePBRTMaterials: false } });
    }
});
