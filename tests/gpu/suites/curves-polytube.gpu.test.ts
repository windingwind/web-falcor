/**
 * Curve tessellation modes (CurveTessellation::convertToPolytube and the
 * `TessellateCurvesIntoPolyTubes` builder flag / `curves:mode` setting).
 *
 * Falcor's two_curves.usda is imported both ways. As linear swept spheres
 * every hit sits at the curve's radius from the centreline. As poly tubes the
 * curve is a triangle mesh with square cross sections whose circumradius is
 * the radius scaled by kMeshCompensationScale (1.11), so every hit must lie
 * between that square's inradius and circumradius — a band the swept-sphere
 * surface never leaves the inside of.
 */

import { RenderGraph, SceneBuilderFlags, createPass, extractBasisCurvesFromUsda, getGlobalSettings, initScripting, kMeshCompensationScale, runSceneScript, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 128;
const kMedia = "/Falcor/media/test_scenes";

async function loadCurves(device: Device, flags?: number): Promise<Scene> {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/curves-view.pyscene")).text();
    const scene = await runSceneScript(device, source, kMedia, flags !== undefined ? { flags } : undefined);
    scene.camera.setAspectRatio(1);
    return scene;
}

async function hitPositions(device: Device, scene: Scene): Promise<[number, number, number][]> {
    const graph = new RenderGraph(device, "Curves");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("GBuffer.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const posW = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBuffer.posW")!)).buffer);
    const hits: [number, number, number][] = [];
    for (let i = 0; i < size * size; i++) if (posW[i * 4 + 3] !== 0) hits.push([posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!]);
    return hits;
}

/** Distance to the nearest centreline segment and the radius interpolated there. */
function nearest(segments: { a: number[]; b: number[]; ra: number; rb: number }[], p: number[]): { d: number; r: number } {
    let best = { d: Infinity, r: 0 };
    for (const s of segments) {
        const ab = [0, 1, 2].map((i) => s.b[i]! - s.a[i]!);
        const ap = [0, 1, 2].map((i) => p[i]! - s.a[i]!);
        const t = Math.min(Math.max((ab[0]! * ap[0]! + ab[1]! * ap[1]! + ab[2]! * ap[2]!) / (ab[0]! ** 2 + ab[1]! ** 2 + ab[2]! ** 2), 0), 1);
        const q = [0, 1, 2].map((i) => s.a[i]! + t * ab[i]!);
        const d = Math.hypot(p[0]! - q[0]!, p[1]! - q[1]!, p[2]! - q[2]!);
        if (d < best.d) best = { d, r: s.ra + t * (s.rb - s.ra) };
    }
    return best;
}

gpuTest("CurvesPolytube.flagTurnsCurvesIntoTubeMeshes", async ({ device }) => {
    const usda = await (await fetch(`${kMedia}/curves/two_curves.usda`)).text();
    const segments: { a: number[]; b: number[]; ra: number; rb: number }[] = [];
    for (const c of extractBasisCurvesFromUsda(usda)) {
        for (let j = 0; j + 1 < c.widths.length; j++) {
            segments.push({ a: [...c.points.slice(j * 3, j * 3 + 3)], b: [...c.points.slice(j * 3 + 3, j * 3 + 6)], ra: c.widths[j]! / 2, rb: c.widths[j + 1]! / 2 });
        }
    }

    const lss = await loadCurves(device);
    expectEq(lss.hasCurves, true, "by default the curves stay swept-sphere curves");
    const tube = await loadCurves(device, SceneBuilderFlags.TessellateCurvesIntoPolyTubes);
    expectEq(tube.hasCurves, false, "with the flag they become meshes");

    const lssHits = await hitPositions(device, lss);
    const tubeHits = await hitPositions(device, tube);
    let lssWorst = 0;
    let outsideBand = 0;
    let checked = 0;
    for (const p of lssHits) {
        const { d, r } = nearest(segments, p);
        if (d > 1) continue; // the tiny helper triangle in the file
        lssWorst = Math.max(lssWorst, Math.abs(d - r));
    }
    for (const p of tubeHits) {
        const { d, r } = nearest(segments, p);
        if (d > 1) continue;
        checked++;
        const circum = r * kMeshCompensationScale;
        // The band between the square cross section's inradius and circumradius,
        // with slack for the chord between rings of different radius.
        if (d < circum / Math.SQRT2 - 0.01 || d > circum + 0.01) outsideBand++;
    }
    console.error(`# curves: LSS ${lssHits.length} hits, worst |d - r| ${lssWorst.toExponential(2)}; polytube ${tubeHits.length} hits, ${outsideBand}/${checked} outside the tube band`);
    expectEq(lssHits.length > 200 && tubeHits.length > 200, true, "both curves are on screen");
    expectEq(lssWorst < 5e-3, true, `swept-sphere hits sit at the curve radius (${lssWorst})`);
    expectEq(outsideBand, 0, "every polytube hit lies on the tube");
});

gpuTest("CurvesPolytube.settingsModeOverridesTheFlag", async ({ device }) => {
    // Native lets a per-prim `curves:mode` attribute pick the mode either way.
    const settings = getGlobalSettings();
    settings.addFilteredAttributes({ regex: ".*/curve0", attributes: { curves: { mode: "polytube" } } });
    try {
        const scene = await loadCurves(device);
        expectEq(scene.hasCurves, true, "curve1 keeps the default swept-sphere mode");
        const hits = await hitPositions(device, scene);
        expectEq(hits.length > 200, true, "the mixed scene still renders");
    } finally {
        settings.clearFilteredAttributes();
    }
});
