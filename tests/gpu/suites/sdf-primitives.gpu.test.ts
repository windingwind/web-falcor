/**
 * SDF grids built from a `.sdf` primitive list (SDFGrid::loadPrimitivesFromFile):
 * CSG shapes defined by transformed spheres/boxes folded together with union,
 * subtraction and their smooth variants.
 *
 * The fixture from `node scripts/gen-assets.mjs sdf` is a sphere unioned with a
 * bar rotated 45° about z, minus a sphere at the top, so the rendered surface
 * has a closed form: every hit must sit on the zero level set of that CSG
 * expression, the bar must lie on one diagonal, and the subtracted sphere must
 * leave a hole where the union alone would have been solid.
 *
 * Web divergence (docs §9): native evaluates the primitives on the GPU and only
 * SDFSBS accepts them; the port bakes the same kernel's corner values host-side,
 * which feeds every grid type.
 */

import { RenderGraph, createPass, evalSDFPrimitive, initScripting, loadSDFPrimitives, runSceneScript, SDF3DShapeType, SDFOperationType, type SDF3DPrimitive } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose, SkipError } from "../harness/registry.js";

const size = 128;
/** Largest float32: what EvaluateSDFPrimitives.cs.slang starts each sample from. */
const kFltMax = 3.402823466e38;

/** The CSG expression the fixture encodes, folded in file order. */
function evalCSG(primitives: SDF3DPrimitive[], p: [number, number, number]): number {
    let d = kFltMax;
    for (const primitive of primitives) d = evalSDFPrimitive(primitive, p, d);
    return d;
}

gpuTest("SDFFromPrimitives.csgSurfaceMatchesThePrimitiveList", async ({ device }) => {
    if (!(await fetch("/Falcor/media/sdf/sdf-primitives.sdf", { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/sdf/sdf-primitives.sdf missing (node scripts/gen-assets.mjs sdf)");
    }
    const primitives = await loadSDFPrimitives("/Falcor/media/sdf/sdf-primitives.sdf");
    expectEq(primitives.length, 3, "primitives in the file");

    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/sdf-from-primitives.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media");
    scene.camera.setAspectRatio(1.0);
    expectEq(scene.sdfGrids.length, 1, "one SDF grid");

    const graph = new RenderGraph(device, "SDFPrimitives");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBufferRT");
    graph.markOutput("GBufferRT.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const posW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.posW")!)).buffer);

    const errors: number[] = [];
    let barHits = 0;
    let mirroredHits = 0;
    let dentHits = 0;
    for (let i = 0; i < size * size; i++) {
        if (posW[i * 4 + 3] === 0) continue; // background
        const p: [number, number, number] = [posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!];
        errors.push(Math.abs(evalCSG(primitives, p)));
        // The bar runs along the +45° diagonal; nothing reaches that far out on
        // the other one, so a dropped transpose would swap these two counts.
        if (p[0] > 0.22 && p[1] > 0.22) barHits++;
        if (p[0] > 0.22 && p[1] < -0.22) mirroredHits++;
        // Inside the subtracted sphere's footprint, the surface is the dent's
        // wall, well below the union's front face at z = sqrt(0.28^2 - r^2).
        if (Math.hypot(p[0], p[1]) < 0.08 && p[2] < 0.2) dentHits++;
    }
    errors.sort((a, b) => a - b);
    const hits = errors.length;
    const median = errors[hits >> 1] ?? 0;
    const p99 = errors[Math.min(hits - 1, Math.floor(hits * 0.99))] ?? 0;
    console.error(`# SDF from primitives: ${hits}/${size * size} hits, |CSG| median ${median.toExponential(2)}, p99 ${p99.toExponential(2)}, max ${(errors[hits - 1] ?? 0).toExponential(2)}`);
    console.error(`# bar hits ${barHits}, mirrored-diagonal hits ${mirroredHits}, dent hits ${dentHits}`);

    expectEq(hits > 500, true, `the CSG solid is visible (${hits} hits)`); // the solid covers ~6% of the frame
    // A 64^3 grid resolves the surface to about one voxel (1/64 = 0.016); the
    // sharp box edges are where trilinear interpolation deviates most.
    expectClose(median, 0, 5e-3, "hits lie on the analytic CSG surface");
    expectEq(p99 < 0.01, true, `99% of hits are within a voxel (${p99})`);
    expectEq(barHits > 20, true, `the rotated bar is on the +45° diagonal (${barHits} hits)`);
    expectEq(mirroredHits, 0, "nothing is on the mirrored diagonal");
    expectEq(dentHits > 20, true, `the subtracted sphere carved a dent (${dentHits} hits)`);
});

gpuTest("SDFFromPrimitives.runtimeEditsRebakeTheGrid", async ({ device }) => {
    if (!(await fetch("/Falcor/media/sdf/sdf-primitives.sdf", { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/sdf/sdf-primitives.sdf missing (node scripts/gen-assets.mjs sdf)");
    }
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/tests/oracle/assets/sdf-from-primitives.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media");
    scene.camera.setAspectRatio(1.0);

    const editor = scene.sdfGrids[0]!.grid.primitives!;
    expectEq(editor.primitiveCount, 3, "the file's primitives are still on the grid");

    const graph = new RenderGraph(device, "SDFEdit");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBufferRT");
    graph.markOutput("GBufferRT.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;

    /** Renders and reports the hits, plus how many land in the empty corner. */
    const readHits = async (): Promise<{ hits: [number, number, number][]; corner: number }> => {
        graph.execute(ctx);
        const posW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRT.posW")!)).buffer);
        const hits: [number, number, number][] = [];
        let corner = 0;
        for (let i = 0; i < size * size; i++) {
            if (posW[i * 4 + 3] === 0) continue;
            const p: [number, number, number] = [posW[i * 4]!, posW[i * 4 + 1]!, posW[i * 4 + 2]!];
            hits.push(p);
            // The anti-diagonal corner the fixture leaves empty.
            if (p[0] < -0.2 && p[1] > 0.2) corner++;
        }
        return { hits, corner };
    };

    const before = await readHits();
    expectEq(before.corner, 0, "nothing sits on the anti-diagonal to start with");

    // Add a blob where the fixture is empty, then re-bake and re-render.
    const blob: SDF3DPrimitive = {
        shapeType: SDF3DShapeType.Sphere,
        shapeData: [0.12, 0, 0],
        shapeBlobbing: 0,
        operationType: SDFOperationType.Union,
        operationSmoothing: 0,
        translation: [-0.35, 0.35, 0],
        invRotationScale: [1, 0, 0, 0, 1, 0, 0, 0, 1],
    };
    const blobID = editor.addPrimitives([blob]);
    expectEq(scene.updateSDFGrids(), true, "the edited grid re-bakes");
    expectEq(scene.updateSDFGrids(), false, "a clean grid does not re-bake again");

    const added = await readHits();
    const withBlob = [...(await loadSDFPrimitives("/Falcor/media/sdf/sdf-primitives.sdf")), blob];
    let worst = 0;
    for (const p of added.hits) worst = Math.max(worst, Math.abs(evalCSG(withBlob, p)));
    console.error(`# SDF runtime edit: ${before.hits.length} -> ${added.hits.length} hits, ${added.corner} on the new blob, worst |CSG| ${worst.toExponential(2)}`);
    expectEq(added.corner > 20, true, `the added primitive is on screen (${added.corner} hits)`);
    expectEq(added.hits.length > before.hits.length, true, "the edited surface covers more of the frame");
    expectEq(worst < 0.02, true, `hits lie on the edited CSG surface (${worst})`);

    // Removing it must put the surface back exactly where it was.
    editor.removePrimitives([blobID]);
    expectEq(scene.updateSDFGrids(), true, "the removal re-bakes too");
    const after = await readHits();
    expectEq(after.corner, 0, "the blob is gone");
    expectEq(after.hits.length, before.hits.length, "the original surface is restored");
});
