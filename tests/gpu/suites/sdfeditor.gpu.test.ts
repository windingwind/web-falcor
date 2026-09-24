/**
 * SDFEditor over the upstream two-grid editor scene with cheese grids
 * (assets/sdfeditor-two-grids.pyscene; two distinct SBS grids, packed into one binding).
 *
 * 1. GUI layer vs native: native's SBS surface is unusable on this machine (docs/testing.md), so
 *    the editor runs over native's own input (its ToneMapper.dst capture) instead of the web
 *    render; bounding boxes, the current-mode badge and the rest of the 2D/3D GUI must then
 *    reproduce native's SDFEditor.output.
 * 2. The unmodified upstream graph (SDFEditorRenderGraphV2.py) renders both grids, and an
 *    Alt+LMB edit through the pass's key/mouse events adds a primitive to the selected grid
 *    and changes the image; Ctrl+Z removes it again.
 *
 * Regenerate: Mogwai --script tests/oracle/render-native-sdfeditor.py --headless
 */

import { RenderGraph, createPass, float4, initScripting, mulMatVec, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import type { SDFEditor } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const [w, h] = [640, 360];
const frames = 64;
const kScene = "/tests/oracle/assets/sdfeditor-two-grids.pyscene";

const srgb = (v: number) => {
    const c = Math.min(Math.max(v, 0), 1);
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
};

gpuTest("SDFEditor.guiMatchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const scene = await runSceneScript(device, await (await fetch(kScene)).text(), "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    const graph = new RenderGraph(device, "SDFEditorGUI");
    graph.addPass(createPass(device, "GBufferRT", {}), "GBufferRT");
    graph.addPass(createPass(device, "ImageLoader", { filename: `/tests/oracle/out-native/sdfeditor.ToneMapper.dst.${frames}.exr`, srgb: false }), "Input");
    graph.addPass(createPass(device, "SDFEditor", {}), "SDFEditor");
    graph.addEdge("GBufferRT.vbuffer", "SDFEditor.vbuffer");
    graph.addEdge("GBufferRT.linearZ", "SDFEditor.linearZ");
    graph.addEdge("Input.dst", "SDFEditor.inputColor");
    graph.markOutput("SDFEditor.output");
    graph.markOutput("Input.dst");
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("SDFEditor.output")!)).buffer);
    const input = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("Input.dst")!)).buffer);

    const bitmap = await createImageBitmap(await (await fetch(`/tests/oracle/out-native/sdfeditor.SDFEditor.output.${frames}.png`)).blob(), { colorSpaceConversion: "none" });
    expectEq([bitmap.width, bitmap.height].join("x"), `${w}x${h}`, "oracle resolution");
    const c2d = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })!;
    c2d.drawImage(bitmap, 0, 0);
    const nat = c2d.getImageData(0, 0, w, h).data;
    // Native's PNG capture sRGB-encodes the float output; GUI pixels are where native changed its input.
    // The edit-primitive preview blends at 0.5, or 0.1 behind the scene depth (GUIPass hitAlpha /
    // hiddenHitAlpha): web's cheese surface hides it where native's (no SBS hits) does not.
    const lin = (v: number) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    let [gui, guiBad, hiddenPreview, bad] = [0, 0, 0, 0];
    for (let i = 0; i < w * h; i++) {
        let d = 0;
        let g = 0;
        let hidden = 0;
        for (let c = 0; c < 3; c++) {
            const [x, n] = [input[i * 4 + c]!, lin(nat[i * 4 + c]! / 255)];
            d = Math.max(d, Math.abs(srgb(web[i * 4 + c]!) - nat[i * 4 + c]! / 255));
            g = Math.max(g, Math.abs(srgb(x) - nat[i * 4 + c]! / 255));
            const previewColor = (n - 0.5 * x) / 0.5;
            hidden = Math.max(hidden, Math.abs(srgb(0.9 * x + 0.1 * previewColor) - srgb(web[i * 4 + c]!)));
        }
        if (g <= 4 / 255) continue;
        gui++;
        if (d <= 0.1) continue;
        if (hidden <= 4 / 255) hiddenPreview++;
        else guiBad++;
    }
    for (let i = 0; i < w * h; i++) {
        let d = 0;
        for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(srgb(web[i * 4 + c]!) - nat[i * 4 + c]! / 255));
        if (d > 0.1) bad++;
    }
    console.error(`# sdfeditor gui: ${gui} GUI px, ${guiBad} off, ${hiddenPreview} hidden-preview px; ${bad} off overall`);
    expectEq(gui > 3000, true, `native GUI footprint ${gui} px`);
    expectEq(guiBad <= gui * 0.01, true, `GUI pixels off: ${guiBad} of ${gui}`);
    expectEq(bad <= guiBad + hiddenPreview, true, `pixels off outside the GUI: ${bad - guiBad - hiddenPreview}`);
});

gpuTest("SDFEditor.upstreamGraphRendersAndEdits", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const [graph] = await runGraphScript(device, await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/SDFEditorRenderGraphV2.py")).text());
    const scene = await runSceneScript(device, await (await fetch(kScene)).text(), "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    graph!.onResize(w, h);
    graph!.setScene(scene);
    graph!.markOutput("GBufferRT.vbuffer");
    await graph!.init();
    for (let f = 0; f < 4; f++) graph!.execute(ctx);
    const readVBuffer = async () => new Uint32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.vbuffer")!)).buffer);

    // Both grid instances are hit (HitInfo header: 3 type bits, SDFGrid = 6).
    const vbuffer = await readVBuffer();
    const ids = scene.getSDFGridInstanceIDs();
    const hits = ids.map(() => 0);
    const primBits = Number(scene.getSceneDefines().get("HIT_INFO_PRIMITIVE_INDEX_BITS"));
    for (let i = 0; i < w * h; i++) {
        const header = vbuffer[i * 4]!;
        if (header >>> 29 === 6) hits[ids.indexOf((header & ((1 << 29) - 1)) >>> primBits)]!++;
    }
    console.error(`# sdfeditor grid hits: ${hits.join(", ")}; ids ${ids}, primitive bits ${primBits}`);
    expectEq(hits.every((n) => n > 5000), true, `hits per grid ${hits}`);

    // Alt+LMB on the selected grid's surface (picking readback lags a frame).
    const editor = graph!.getPass("SDFEditor") as SDFEditor;
    const gridID = scene.findSDFGridIDFromGeometryInstanceID(ids[0]!);
    const grid = scene.sdfGrids[gridID]!.grid;
    const t = scene.getSDFGridTransform(gridID);
    const clip = mulMatVec(scene.camera.getViewProjMatrix(), new float4(t.get(0, 3), t.get(1, 3), t.get(2, 3), 1));
    const target: [number, number] = [0.5 + (0.5 * clip.x) / clip.w, 0.5 - (0.5 * clip.y) / clip.w];
    editor.onMouseEvent({ type: "move", pos: target });
    graph!.execute(ctx);
    const before = await readVBuffer();
    const count0 = grid.primitives?.primitiveCount ?? 0;
    editor.onKeyEvent({ type: "keyPressed", key: "LeftAlt", mods: { alt: true } });
    editor.onMouseEvent({ type: "move", pos: target });
    editor.onMouseEvent({ type: "buttonDown", button: "left", pos: target });
    editor.onMouseEvent({ type: "buttonUp", button: "left", pos: target });
    editor.onKeyEvent({ type: "keyReleased", key: "LeftAlt" });
    const count1 = grid.primitives?.primitiveCount ?? 0;
    expectEq(count1, count0 + 1, "Alt+LMB adds one primitive");
    for (let f = 0; f < 2; f++) graph!.execute(ctx);
    // The V-buffer is deterministic: the added sphere changes the hits around the cursor only.
    const after = await readVBuffer();
    let changed = 0;
    for (let i = 0; i < w * h * 4; i++) if (after[i] !== before[i]) changed++;
    console.error(`# sdfeditor edit: primitives ${count0} -> ${count1}, ${changed} V-buffer words changed`);
    expectEq(changed > 20 && changed < 20000, true, `edit changes the hits near the cursor (${changed} words)`);

    editor.onKeyEvent({ type: "keyPressed", key: "LeftControl", mods: { ctrl: true } });
    editor.onKeyEvent({ type: "keyPressed", key: "Z", mods: { ctrl: true } });
    editor.onKeyEvent({ type: "keyReleased", key: "Z", mods: { ctrl: true } });
    editor.onKeyEvent({ type: "keyReleased", key: "LeftControl" });
    expectEq(grid.primitives?.primitiveCount ?? 0, count0, "Ctrl+Z removes it");
});

gpuTest("SDFEditor.upstreamSceneRenders", async ({ device }) => {
    // The unmodified upstream image test: SDFEditorSceneTwoSDFs.pyscene (a value-file grid and a
    // 256^3 primitive-list grid) through SDFEditorRenderGraphV2.py.
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const dir = "/Falcor/tests/image_tests/scene/scenes";
    const t0 = performance.now();
    const [graph] = await runGraphScript(device, await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/SDFEditorRenderGraphV2.py")).text());
    const scene = await runSceneScript(device, await (await fetch(`${dir}/SDFEditorSceneTwoSDFs.pyscene`)).text(), dir);
    scene.camera.setAspectRatio(w / h);
    graph!.onResize(w, h);
    graph!.setScene(scene);
    graph!.markOutput("GBufferRT.vbuffer");
    await graph!.init();
    graph!.execute(ctx);
    const vbuffer = new Uint32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.vbuffer")!)).buffer);
    const ids = scene.getSDFGridInstanceIDs();
    const primBits = Number(scene.getSceneDefines().get("HIT_INFO_PRIMITIVE_INDEX_BITS"));
    const hits = ids.map(() => 0);
    for (let i = 0; i < w * h; i++) {
        const header = vbuffer[i * 4]!;
        if (header >>> 29 === 6) hits[ids.indexOf((header & ((1 << 29) - 1)) >>> primBits)]!++;
    }
    console.error(`# sdfeditor upstream scene: ${(performance.now() - t0).toFixed(0)} ms, grid hits ${hits.join(", ")}`);
    expectEq(hits.every((n) => n > 1000), true, `hits per grid ${hits}`);
});
