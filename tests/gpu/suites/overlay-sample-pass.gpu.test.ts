/**
 * OverlaySamplePass: the input passes through, and RenderGraph.renderOverlayUI draws native's
 * 5x3 primitive grid (Mogwai's overlay canvas; an OffscreenCanvas here) at the frame size.
 */

import { OverlayDrawList, RenderGraph, createPass, initScripting } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("OverlaySamplePass.drawsPrimitiveGrid", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const [W, H] = [1920, 1080];
    const g = new RenderGraph(device, "OverlayTest");
    g.addPass(createPass(device, "ImageLoader", { filename: "test_scenes/textures/checker_tile_base_color.png", mips: false, srgb: false, outputFormat: "RGBA32Float" }), "ImageLoader");
    g.addPass(createPass(device, "OverlaySamplePass", {}), "Overlay");
    g.addEdge("ImageLoader.dst", "Overlay.input");
    g.markOutput("Overlay.output");
    g.onResize(W, H);
    await g.init();
    const ctx = device.renderContext;
    const canvas = new OffscreenCanvas(W, H);
    const drawList = new OverlayDrawList(canvas.getContext("2d")!);
    g.renderOverlayUI(drawList);
    const px0 = drawList.ctx.getImageData(0, 0, 1, 1).data[3];
    expectEq(px0, 0, "nothing drawn before the graph executes");
    g.execute(ctx);
    const out = new Float32Array((await ctx.readTextureSubresource(g.getOutput("Overlay.output")!)).buffer);
    let nonzero = 0;
    for (let i = 0; i < out.length; i += 4) if (out[i]! > 0) nonzero++;
    expectEq(nonzero > W * H * 0.3, true, `input copied (${nonzero} px)`);

    g.renderOverlayUI(drawList);
    const at = (x: number, y: number) => Array.from(drawList.ctx.getImageData(x, y, 1, 1).data);
    // Cell (i, j) outer rect and the inner area its primitive fills, as native computes them.
    const cell = (i: number, j: number) => {
        const [fw, fh] = [W - 100, H - 100];
        const r = [50 + (fw * i) / 5 + 50, 50 + (fh * j) / 3 + 50, 50 + (fw * (i + 1)) / 5 - 50, 50 + (fh * (j + 1)) / 3 - 50];
        return { rect: r, inner: [r[0]! + 50, r[1]! + 50, r[2]! - 50, r[3]! - 50] };
    };
    const c0 = cell(0, 0);
    const edge = at(Math.floor(c0.rect[0]!), Math.round((c0.rect[1]! + c0.rect[3]!) / 2));
    expectEq(edge[3]! > 200 && edge[0]! > 200, true, `cell outline is white (${edge})`);
    const frame = at(50, H / 2);
    expectEq(frame[3], 0, "the alpha-0 red frame is skipped, as ImGui does");
    const circle = cell(3, 1).inner; // primType 8: filled circle
    const cc = at(Math.round((circle[0]! + circle[2]!) / 2), Math.round((circle[1]! + circle[3]!) / 2));
    expectEq(cc.join(","), "255,255,255,255", "filled circle center");
    const hollow = cell(2, 1).inner; // primType 7: circle outline, empty inside
    expectEq(at(Math.round((hollow[0]! + hollow[2]!) / 2), Math.round((hollow[1]! + hollow[3]!) / 2))[3], 0, "circle outline is hollow");
    const grad = cell(2, 0).inner; // primType 2: UL white, UR blue, BR green, BL red
    const ur = at(Math.ceil(grad[2]!) - 2, Math.ceil(grad[1]!) + 1);
    const bl = at(Math.ceil(grad[0]!) + 1, Math.floor(grad[3]!) - 2);
    expectEq(ur[2]! > 230 && ur[0]! < 30 && bl[0]! > 230 && bl[2]! < 30, true, `gradient corners (UR ${ur}, BL ${bl})`);
    const text = cell(1, 2).inner; // primType 11: "Hello, world!"
    const tx = drawList.ctx.getImageData(Math.floor(text[0]!), Math.floor(text[1]!), 100, 16).data;
    let ink = 0;
    for (let i = 3; i < tx.length; i += 4) if (tx[i]! > 128) ink++;
    console.error(`# overlay: edge ${edge}, gradient UR ${ur} BL ${bl}, text ink ${ink} px`);
    expectEq(ink > 50, true, `text drawn (${ink} px)`);
});
