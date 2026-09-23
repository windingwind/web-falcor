/**
 * Graph output channel masks (RenderGraph::markOutput(name, mask)) and Mogwai
 * FrameCapture: one file per mask, single channels extracted, RGBA with alpha.
 */

import { Bitmap, Properties, RenderData, RenderGraph, RenderPass, RenderPassReflection, ResourceFormat, TextureChannelFlags, type CompileData, type Device, type RenderContext } from "@web-falcor/falcor";
import { captureOutput } from "../../../packages/mogwai/src/FrameCapture.js";
import { gpuTest, expectEq } from "../harness/registry.js";

class Constant extends RenderPass {
    constructor(device: Device, _props: Properties) {
        super(device);
    }
    override reflect(_cd: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("color", "constant").format(ResourceFormat.RGBA32Float);
        r.addOutput("extra", "second output").format(ResourceFormat.RGBA32Float);
        return r;
    }
    override execute(ctx: RenderContext, rd: RenderData): void {
        ctx.clearTexture(rd.getTexture("color")!, [0.25, 0.5, 0.75, 0.125]);
        ctx.clearTexture(rd.getTexture("extra")!, [1, 1, 1, 1]);
    }
}

gpuTest("FrameCapture.outputMasks", async ({ device }) => {
    const graph = new RenderGraph(device, "Masks");
    graph.addPass(new Constant(device, new Properties()), "C");
    graph.markOutput("C.color");
    graph.markOutput("C.color", TextureChannelFlags.Green);
    graph.markOutput("C.color", TextureChannelFlags.RGBA);
    expectEq(graph.getOutputNames().join(), "C.color", "re-marking merges masks instead of duplicating the output");
    expectEq([...graph.getOutputMasks(0)].sort((a, b) => a - b).join(), "2,7,15", "the output keeps all three masks");
    expectEq(graph.getAvailableOutputs().join(), "C.color,C.extra", "available outputs");
    expectEq(graph.exportScript().includes('g.markOutput("C.color", TextureChannelFlags.Green)'), true, "the exporter writes masks");

    graph.onResize(8, 4);
    await graph.init();
    graph.execute(device.renderContext);
    const files = await captureOutput(device, graph, 0, "frame", false);
    expectEq(files.map((f) => f.name).sort().join(), "frame.G.exr,frame.RGBA.exr,frame.exr", "one file per mask, with native suffixes");

    const decode = async (name: string) => (await Bitmap.createFromBytes(files.find((f) => f.name === name)!.bytes, name, true))!;
    const rgb = await decode("frame.exr");
    const g = await decode("frame.G.exr");
    const rgba = await decode("frame.RGBA.exr");
    const px = (b: Bitmap) => Array.from(new Uint16Array(b.data.slice(0, 8).buffer)).map((h) => Number(h.toString()));
    // All-half EXRs decode to RGBA16Float; compare through float conversion.
    const f16 = (b: Bitmap) => px(b).map((h) => {
        const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
        return s * (e === 0 ? m / 1024 * 2 ** -14 : (1 + m / 1024) * 2 ** (e - 15));
    });
    expectEq(f16(rgb).join(), "0.25,0.5,0.75,1", "RGB capture drops alpha");
    // One-channel float captures widen like native's blit to RGBA32Float: (g, 0, 0, 1).
    expectEq(f16(g).join(), "0.5,0,0,1", "the G mask copies green into a single-channel image");
    expectEq(f16(rgba).join(), "0.25,0.5,0.75,0.125", "RGBA keeps alpha");

    graph.markOutput("*");
    expectEq(graph.getOutputNames().join(), "C.color,C.extra", "'*' marks every available output");
});
