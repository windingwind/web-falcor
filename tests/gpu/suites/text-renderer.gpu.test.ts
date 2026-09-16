/**
 * TextRenderer port (upstream font atlas + TextRenderer.3d.slang) through the
 * ComparisonPass "showTextLabels" option: labels land in the native positions
 * (row defaultTexDims.y - 32, 16 px either side of the divider) and nothing
 * else in the image changes.
 */

import { RenderGraph, ResourceFormat, TextRenderer, createPass } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("TextRenderer.splitScreenLabels", async ({ device }) => {
    const w = 256;
    const h = 128;
    const solid = (r: number, g: number, b: number) => {
        const data = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) data.set([r, g, b, 1], i * 4);
        return device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, data);
    };
    const left = solid(0.2, 0.2, 0.2);
    const right = solid(0.2, 0.2, 0.2);
    const ctx = device.renderContext;

    const render = async (showTextLabels: boolean) => {
        const g = new RenderGraph(device, "Split");
        g.addPass(createPass(device, "SplitScreenPass", { showTextLabels, leftLabel: "LEFT", rightLabel: "RIGHT", splitLocation: 0.5 }), "Pass");
        g.setInput("Pass.leftInput", left);
        g.setInput("Pass.rightInput", right);
        g.markOutput("Pass.output");
        g.onResize(w, h);
        await g.init(); // waits for the font atlas when labels are on
        g.execute(ctx);
        return new Float32Array((await ctx.readTextureSubresource(g.getOutput("Pass.output")!)).buffer);
    };
    const plain = await render(false);
    const labeled = await render(true);

    // Native placement: y = h - 32 (glyphs ~17 px tall), right label at x >= splitX + 16,
    // left label ending at x <= splitX - 16. Shadow adds one pixel down/right.
    const splitX = w / 2;
    let insideLeft = 0;
    let insideRight = 0;
    let outside = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            const i = (y * w + x) * 4;
            const changed = Math.abs(labeled[i]! - plain[i]!) > 1e-3 || Math.abs(labeled[i + 1]! - plain[i + 1]!) > 1e-3;
            if (!changed) continue;
            const inRow = y >= h - 32 && y <= h - 32 + 18;
            if (inRow && x >= splitX + 16 && x < splitX + 16 + 5 * 9 + 1) insideRight++;
            else if (inRow && x >= splitX - 16 - 4 * 9 && x < splitX - 16 + 1) insideLeft++;
            else outside++;
        }
    }
    console.error(`# labels: left=${insideLeft} right=${insideRight} outside=${outside}`);
    expectEq(insideLeft > 20, true, `left label drawn (${insideLeft} px)`);
    expectEq(insideRight > 20, true, `right label drawn (${insideRight} px)`);
    expectEq(outside, 0, "no pixels changed outside the label boxes");

    // Text is white on top of a black shadow: some label pixels must be brighter than the background.
    let bright = 0;
    for (let i = 0; i < w * h; i++) if (labeled[i * 4]! > 0.9) bright++;
    expectEq(bright > 20, true, `white glyph pixels (${bright})`);
    expectEq(new TextRenderer(device).isReady(), false, "font loads asynchronously");
});
