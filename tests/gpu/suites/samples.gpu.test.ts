/**
 * Falcor's sample applications (packages/samples) on the web SampleApp, run headless on
 * the harness device: each renders frames into its target FBO, checked for the content
 * the native sample draws.
 */

import { type Device, type SampleApp, type SampleAppConfig } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";
import { SampleAppTemplate } from "../../../packages/samples/src/SampleAppTemplate.js";
import { ShaderToy } from "../../../packages/samples/src/ShaderToy.js";
import { Visualization2D, Visualization2DScene } from "../../../packages/samples/src/Visualization2D.js";
import { MultiSampling } from "../../../packages/samples/src/MultiSampling.js";

async function start<T extends SampleApp>(device: Device, Sample: new (config: SampleAppConfig) => T, width = 128, height = 128): Promise<T> {
    const app = new Sample({ device, headless: true, windowDesc: { width, height } });
    await app.initialize();
    return app;
}

/** The target FBO's color as RGBA bytes (BGRA8 targets are swizzled back). */
async function readTarget(app: SampleApp): Promise<Uint8Array> {
    const tex = app.getTargetFbo().getColorTexture(0)!;
    const bytes = await app.getRenderContext().readTextureSubresource(tex, 0);
    if (!tex.gpuFormat.startsWith("bgra")) return bytes;
    const out = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i += 4) out.set([bytes[i + 2]!, bytes[i + 1]!, bytes[i]!, bytes[i + 3]!], i);
    return out;
}

const srgb = (v: number) => Math.round(255 * (v <= 0.0031308 ? v * 12.92 : 1.055 * Math.pow(v, 1 / 2.4) - 0.055));
const distinct = (px: Uint8Array) => new Set(Array.from({ length: px.length / 4 }, (_, i) => px[i * 4]! | (px[i * 4 + 1]! << 8) | (px[i * 4 + 2]! << 16))).size;

gpuTest("Samples.sampleAppTemplateClearsTheFrame", async ({ device }) => {
    const app = await start(device, SampleAppTemplate);
    app.renderFrame();
    const px = await readTarget(app);
    const want = [srgb(0.38), srgb(0.52), srgb(0.1), 255];
    let bad = 0;
    for (let i = 0; i < px.length; i += 4) if (Math.max(...want.map((w, c) => Math.abs(px[i + c]! - w))) > 1) bad++;
    expectEq(bad, 0, `pixels off the clear color ${want}`);
});

gpuTest("Samples.shaderToyAnimatesWithTheClock", async ({ device }) => {
    const app = await start(device, ShaderToy);
    const clock = app.getGlobalClock();
    const at = async (t: number) => {
        clock.pause().setTime(t);
        app.renderFrame();
        return readTarget(app);
    };
    const a = await at(1.5);
    const b = await at(1.5);
    const c = await at(4.0);
    let same = 0;
    let moved = 0;
    for (let i = 0; i < a.length; i++) {
        if (a[i] === b[i]) same++;
        if (a[i] !== c[i]) moved++;
    }
    console.error(`# samples ShaderToy: distinct colors ${distinct(a)}, changed bytes over time ${moved}`);
    expectEq(same, a.length, "same time renders the same frame");
    expectEq(distinct(a) > 50 && moved > 1000, true, "the toy is a varied, animated image");
});

gpuTest("Samples.visualization2DFollowsTheMouse", async ({ device }) => {
    const app = await start(device, Visualization2D, 256, 256);
    app.getGlobalClock().pause().setTime(0.5);
    for (const scene of [Visualization2DScene.MarkerDemo, Visualization2DScene.VoxelNormals]) {
        app.selectScene(scene);
        app.mousePosition = [40, 40];
        app.renderFrame();
        const a = await readTarget(app);
        app.mousePosition = [200, 180];
        app.renderFrame();
        const b = await readTarget(app);
        let moved = 0;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) moved++;
        console.error(`# samples Visualization2D ${Visualization2DScene[scene]}: distinct colors ${distinct(a)}, changed bytes after a mouse move ${moved}`);
        expectEq(distinct(a) > 4 && moved > 0, true, `${Visualization2DScene[scene]} draws and reacts to the mouse`);
    }
});

gpuTest("Samples.multiSamplingResolvesTheDisk", async ({ device }) => {
    const app = await start(device, MultiSampling);
    for (const frame of ["resolve + blit", "blit from the multisampled texture"]) {
        // A sentinel target, so a frame that fails to draw can't pass on the previous one.
        app.getRenderContext().clearFbo(app.getTargetFbo(), [1, 0, 1, 1], 1, 0);
        device.gpuDevice.pushErrorScope("validation");
        app.renderFrame();
        const error = await device.gpuDevice.popErrorScope();
        expectEq(error?.message ?? "", "", `${frame}: WebGPU validation`);
        const px = await readTarget(app);
        const at = (x: number, y: number) => px[(y * 128 + x) * 4]!;
        // Disk of radius 0.75 in NDC at gray 0.5 over black; edges get partial coverage.
        const edge = new Set<number>();
        for (let i = 0; i < px.length; i += 4) if (px[i]! !== 0 && px[i]! !== srgb(0.5)) edge.add(px[i]!);
        console.error(`# samples MultiSampling ${frame}: center ${at(64, 64)}, corner ${at(2, 2)}, partial-coverage levels ${[...edge].sort((a, b) => a - b)}`);
        expectEq(at(64, 64), srgb(0.5), `${frame}: disk interior`);
        expectEq(at(2, 2), 0, `${frame}: background`);
        expectEq(edge.size >= 2, true, `${frame}: antialiased edges`);
    }
});
