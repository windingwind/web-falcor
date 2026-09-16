/**
 * Native pass options completed on the web: ToneMapper white balance / exposure
 * value and AccumulatePass max-frame overflow (Stop / Reset / EMA), checked
 * against CPU-computed expectations through real graphs.
 */

import {
    Properties,
    RenderData,
    RenderGraph,
    initScripting,
    runSceneScript,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    createPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { calculateWhiteBalanceTransformRGB_Rec709, invertMat3, mulMat3Vec } from "../../../packages/render-passes/src/ToneMapper/ColorUtils.js";
import { gpuTest, expectEq, expectClose, expectArrayClose } from "../harness/registry.js";

/** Emits a constant colour that changes every frame (frame k -> k/10). */
class FrameColor extends RenderPass {
    frame = 0;
    constructor(device: Device, _props: Properties) {
        super(device);
    }
    override reflect(_cd: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("color", "frame-indexed constant").format(ResourceFormat.RGBA32Float);
        return r;
    }
    override execute(ctx: RenderContext, rd: RenderData): void {
        const v = this.frame++ / 10;
        ctx.clearTexture(rd.getTexture("color")!, [v, v, v, 1]);
    }
}

async function firstPixel(device: Device, tex: Texture): Promise<Float32Array> {
    const bytes = await device.renderContext.readTextureSubresource(tex);
    return new Float32Array(bytes.buffer, bytes.byteOffset, 4);
}

gpuTest("PassOptions.toneMapperWhiteBalanceAndExposureValue", async ({ device }) => {
    // D65 is preserved exactly at 6500 K; a warm target boosts red over blue.
    const identity = calculateWhiteBalanceTransformRGB_Rec709(6500);
    expectArrayClose(identity, [1, 0, 0, 0, 1, 0, 0, 0, 1], 1e-5, "white balance at 6500 K is the identity");
    // Balancing to a 3000 K source white: the (warm) source white maps to neutral, so a gray input cools down.
    const warm = calculateWhiteBalanceTransformRGB_Rec709(3000);
    const sourceWhite = mulMat3Vec(invertMat3(warm), [1, 1, 1]);
    expectEq(sourceWhite[0] > sourceWhite[2], true, `3000 K source white is warm (${sourceWhite.map((v) => v.toFixed(3)).join(", ")})`);
    expectArrayClose(mulMat3Vec(warm, sourceWhite), [1, 1, 1], 1e-5, "the source white transforms to pure white");
    const gray = mulMat3Vec(warm, [0.5, 0.5, 0.5]);
    expectEq(gray[2] > gray[0], true, `gray cools under a 3000 K balance (${gray.map((v) => v.toFixed(3)).join(", ")})`);

    // End to end: Linear operator, clamp off, white balance on -> output = M * (gray * exposure scale).
    const graph = new RenderGraph(device, "WB");
    const tm = graph.addPass(createPass(device, "ToneMapper", { operator: "Linear", clamp: false, whiteBalance: true, whitePoint: 3000, exposureCompensation: 1, outputFormat: "RGBA32Float" }), "ToneMapper");
    const src = new Texture(device, { type: ResourceType.Texture2D, width: 4, height: 4, format: ResourceFormat.RGBA32Float, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget, mipLevels: 1 });
    device.renderContext.clearTexture(src, [0.5, 0.5, 0.5, 1]);
    graph.setInput("ToneMapper.src", src);
    graph.markOutput("ToneMapper.dst");
    graph.onResize(4, 4);
    graph.execute(device.renderContext);
    const out = await firstPixel(device, graph.getOutput("ToneMapper.dst")!);
    // exposure = 2^EC * (ISO/100)/(shutter * fN^2) with the defaults ISO 100, shutter 1, f/1 -> 2.
    const expected = mulMat3Vec(warm, [1, 1, 1]);
    expectArrayClose([out[0]!, out[1]!, out[2]!], expected, 2e-3, "white-balanced linear output");

    // Exposure value (python property / UI slider): aperture priority keeps the f-number and derives the
    // shutter (EV = log2(shutter * fN^2)). As a construction property it is ignored like native.
    const tmp = tm as unknown as { getExposureValue(): number; setExposureValue(v: number): void };
    expectClose(tmp.getExposureValue(), Math.log2(1 * 1 * 1), 1e-6, "EV of the defaults");
    tmp.setExposureValue(4);
    expectClose(tm.getProperties().get("shutter", 0), 16, 1e-4, "EV 4 at f/1 -> shutter 16");
    expectEq(tm.getProperties().get("whiteBalance", false), true, "whiteBalance round-trips");
});

gpuTest("PassOptions.accumulateMaxFramesOverflowModes", async ({ device }) => {
    const ctx = device.renderContext;
    const run = async (props: Record<string, unknown>, frames: number): Promise<{ value: number; source: FrameColor }> => {
        const graph = new RenderGraph(device, "Acc");
        const source = graph.addPass(new FrameColor(device, new Properties()), "Src") as FrameColor;
        graph.addPass(createPass(device, "AccumulatePass", { precisionMode: "Single", ...props }), "Accumulate");
        graph.addEdge("Src.color", "Accumulate.input");
        graph.markOutput("Accumulate.output");
        graph.onResize(2, 2);
        for (let i = 0; i < frames; i++) graph.execute(ctx);
        // Regression guard for the bind-flag resolution: the source output leaves its flags None while
        // Accumulate.input declares ShaderResource; the merged texture must still be a render target.
        const srcTex = graph.getOutput("Src.color")!;
        expectEq((srcTex.bindFlags & ResourceBindFlags.RenderTarget) !== 0, true, "producer output keeps RenderTarget usage after the input merge");
        return { value: (await firstPixel(device, graph.getOutput("Accumulate.output")!))[0]!, source };
    };
    const mean = (n: number) => Array.from({ length: n }, (_, k) => k / 10).reduce((a, b) => a + b, 0) / n;

    // Unlimited: plain running mean over all 8 frames.
    expectClose((await run({}, 8)).value, mean(8), 1e-5, "unlimited accumulation = mean of 8 frames");
    // Stop: frames beyond the limit leave the accumulated image untouched.
    expectClose((await run({ maxFrameCount: 4, overflowMode: "Stop" }, 8)).value, mean(4), 1e-5, "Stop retains the 4-frame mean");
    // Reset: accumulation restarts at frame 4 -> after 8 frames the mean of frames 4..7 is shown.
    const resetMean = (4 / 10 + 5 / 10 + 6 / 10 + 7 / 10) / 4;
    expectClose((await run({ maxFrameCount: 4, overflowMode: "Reset" }, 8)).value, resetMean, 1e-5, "Reset restarts at the limit");
    // EMA: after the limit every frame blends in with the constant weight 1/(max+1).
    let ema = mean(4);
    for (let k = 4; k < 8; k++) ema = ema + (k / 10 - ema) / 5;
    expectClose((await run({ maxFrameCount: 4, overflowMode: "EMA" }, 8)).value, ema, 1e-5, "EMA blends with weight 1/(max+1)");
});

gpuTest("PassOptions.bsdfViewerSliceMode", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const scene = await runSceneScript(device, await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text(), "/Falcor/media/test_scenes");
    const render = async (viewerMode: string): Promise<Float32Array> => {
        const graph = new RenderGraph(device, "BV");
        graph.addPass(createPass(device, "BSDFViewer", { materialID: 0, viewerMode }), "BSDFViewer");
        graph.markOutput("BSDFViewer.output");
        graph.onResize(64, 64);
        graph.setScene(scene);
        graph.execute(device.renderContext);
        return new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("BSDFViewer.output")!)).buffer);
    };
    const material = await render("Material");
    const slice = await render("Slice");
    let finite = true;
    let sumM = 0;
    let sumS = 0;
    let differ = 0;
    for (let i = 0; i < slice.length; i += 4) {
        finite &&= Number.isFinite(slice[i]!) && Number.isFinite(material[i]!);
        sumM += material[i]!;
        sumS += slice[i]!;
        if (Math.abs(slice[i]! - material[i]!) > 1e-3) differ++;
    }
    expectEq(finite, true, "both viewer modes render finite values");
    expectEq(sumM > 0 && sumS > 0, true, `both modes produce content (material ${sumM.toFixed(2)}, slice ${sumS.toFixed(2)})`);
    expectEq(differ > 64, true, `slice view differs from the shaded sphere (${differ} px)`);
});
