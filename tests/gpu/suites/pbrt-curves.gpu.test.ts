/**
 * pbrt-v4 `curve` shapes (PBRTImporter's curve aggregates): strands collect
 * per (transform, material) and become linear swept spheres, or poly-tube
 * meshes under TessellateCurvesIntoPolyTubes.
 *
 * The fixture's strands are straight vertical lines at x = -1 and x = +1, so a
 * hit's distance from its line is the tube radius there: exactly 0.1 on the
 * left; on the right it runs from 0.15 down to 0.075 (native's width taper).
 */

import { RenderGraph, SceneBuilderFlags, createPass, kMeshCompensationScale, runPbrtScene, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const kBase = "/tests/gpu/assets/pbrt";
const size = 128;

async function load(device: Device, flags?: number): Promise<Scene> {
    const scene = await runPbrtScene(device, await (await fetch(`${kBase}/curves.pbrt`)).text(), kBase, { flags });
    scene.camera.setAspectRatio(1);
    return scene;
}

async function hits(device: Device, scene: Scene): Promise<[number, number, number][]> {
    const graph = new RenderGraph(device, "PbrtCurves");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("GBuffer.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const posW = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBuffer.posW")!)).buffer);
    const out: [number, number, number][] = [];
    for (let i = 0; i < size * size; i++) if (posW[i * 4 + 3] !== 0) out.push([posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!]);
    return out;
}

/** Distance from the strand line and which strand it is. */
function strandDistance(p: [number, number, number]): { left: boolean; d: number } {
    const left = p[0] < 0;
    return { left, d: Math.hypot(p[0] - (left ? -1 : 1), p[2]) };
}

gpuTest("PbrtCurves.strandsBecomeSweptSpheresOrPolyTubes", async ({ device }) => {
    const lss = await load(device);
    expectEq(lss.hasCurves, true, "curves stay curves by default");
    expectEq(lss.stats.materials, 2, "one aggregate per material");

    let leftWorst = 0;
    let rightOutside = 0;
    let leftHits = 0;
    let rightHits = 0;
    for (const p of await hits(device, lss)) {
        // Ignore the rounded end caps; the straight sections have a closed form.
        if (p[1] < 0.2 || p[1] > 2.8) continue;
        const { left, d } = strandDistance(p);
        if (left) {
            leftHits++;
            leftWorst = Math.max(leftWorst, Math.abs(d - 0.1));
        } else {
            rightHits++;
            if (d < 0.075 - 2e-3 || d > 0.15 + 2e-3) rightOutside++;
        }
    }
    console.error(`# pbrt curves (LSS): ${leftHits} left hits, worst |d - 0.1| ${leftWorst.toExponential(2)}; ${rightHits} right hits, ${rightOutside} outside [0.075, 0.15]`);
    expectEq(leftHits > 50 && rightHits > 50, true, "both strands are on screen");
    expectEq(leftWorst < 2e-3, true, `the constant-width strand has radius 0.1 (${leftWorst})`);
    expectEq(rightOutside, 0, "the tapering strand stays within its radius range");

    const tube = await load(device, SceneBuilderFlags.TessellateCurvesIntoPolyTubes);
    expectEq(tube.hasCurves, false, "the flag turns them into meshes");
    let outside = 0;
    let checked = 0;
    for (const p of await hits(device, tube)) {
        if (p[1] < 0.2 || p[1] > 2.8) continue;
        const { left, d } = strandDistance(p);
        if (!left) continue;
        checked++;
        const circum = 0.1 * kMeshCompensationScale;
        if (d < circum / Math.SQRT2 - 2e-3 || d > circum + 2e-3) outside++;
    }
    console.error(`# pbrt curves (poly tube): ${outside}/${checked} left-strand hits outside the square tube band`);
    expectEq(checked > 50, true, "the tube is on screen");
    expectEq(outside, 0, "every tube hit lies on the square tube");
});
