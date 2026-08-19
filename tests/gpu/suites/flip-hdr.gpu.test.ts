/**
 * FLIPPass HDR path: auto-exposure parameters from the reference luminance
 * (async readback, applied next frame — same one-frame latency as native's
 * member-after-cbuffer-write order) and pooled FLIP values via parallel
 * reduction. Expected exposure parameters are recomputed independently on the
 * CPU from the known synthetic luminance distribution; pooled values are
 * cross-checked against a CPU scan of the error map.
 */

import { Properties, RenderData, ResourceBindFlags, ResourceFormat } from "@web-falcor/falcor";
import { FLIPPass, computeMedianMax } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("FLIPHdr.autoExposureAndPooledValues", async ({ device }) => {
    const size = 16;
    const n = size * size;

    // Reference: left half HDR-bright (4,4,4), right half dim (0.5,0.5,0.5) -> known median/max luminance.
    const refData = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
        const v = i % size < size / 2 ? 4 : 0.5;
        refData.set([v, v, v, 1], i * 4);
    }
    // Test image: reference distorted in one quadrant.
    const testData = refData.slice();
    for (let i = 0; i < n / 4; i++) testData[i * 4] = testData[i * 4]! * 0.25;

    const mk = (data?: Float32Array, format = ResourceFormat.RGBA32Float, flags?: number) =>
        device.createTexture2D(size, size, format, 1, 1, data, flags);
    const test = mk(testData);
    const reference = mk(refData);
    const errorMap = mk(undefined, ResourceFormat.RGBA32Float, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    const mkDisplay = () => mk(undefined, ResourceFormat.RGBA8UnormSrgb, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    const rd = new RenderData(
        new Map([
            ["testImage", test],
            ["referenceImage", reference],
            ["errorMap", errorMap],
            ["errorMapDisplay", mkDisplay()],
            ["exposureMapDisplay", mkDisplay()],
        ]),
        [size, size],
    );

    const pass = new FLIPPass(device, new Properties({ isHDR: true, computePooledFLIPValues: true }));
    const ctx = device.renderContext;

    // Frame 1 kicks off the luminance readback; poll until the parameters land.
    pass.execute(ctx, rd);
    for (let i = 0; i < 100 && pass.getExposureParameters().numExposures === 2 && pass.getExposureParameters().startExposure === 0; i++) {
        await new Promise((r) => setTimeout(r, 10));
    }

    // Independent CPU expectation: ACES coefficients, t = 0.85 (FLIPPass.cpp).
    const lum = new Float32Array(n);
    for (let i = 0; i < n; i++) lum[i] = 0.2126 * refData[i * 4]! + 0.7152 * refData[i * 4 + 1]! + 0.0722 * refData[i * 4 + 2]!;
    const [median, max] = computeMedianMax(lum);
    const tm = [0.6 * 0.6 * 2.51, 0.6 * 0.03, 0, 0.6 * 0.6 * 2.43, 0.6 * 0.59, 0.14];
    const t = 0.85;
    const a = tm[0]! - t * tm[3]!;
    const b = tm[1]! - t * tm[4]!;
    const c = tm[2]! - t * tm[5]!;
    const d1 = -0.5 * (b / a);
    const xMax = d1 + Math.sqrt(d1 * d1 - c / a);
    const expStart = Math.log2(xMax / max);
    const expStop = Math.log2(xMax / median);
    const expNum = Math.max(2, Math.ceil(expStop - expStart));

    const p = pass.getExposureParameters();
    expectEq(Math.abs(p.startExposure - expStart) < 1e-5, true, `startExposure ${p.startExposure} == ${expStart}`);
    expectEq(p.numExposures, expNum, `numExposures ${p.numExposures} == ${expNum}`);
    expectEq(Math.abs(p.exposureDelta - (expStop - expStart) / (expNum - 1)) < 1e-5, true, `exposureDelta ${p.exposureDelta}`);

    // Frame 2 renders with the landed parameters — the error map is stable from here on.
    pass.execute(ctx, rd);
    const err = new Float32Array((await ctx.readTextureSubresource(errorMap)).buffer);
    let sum = 0;
    let mn = Infinity;
    let mx = -Infinity;
    let nonZero = 0;
    for (let i = 0; i < n; i++) {
        const v = err[i * 4 + 3]!;
        sum += v;
        mn = Math.min(mn, v);
        mx = Math.max(mx, v);
        if (v > 1e-4) nonZero++;
    }
    expectEq(nonZero > 0, true, `distorted quadrant produces FLIP error (${nonZero} px)`);

    // The first reduction may have sampled the frame-1 (default-exposure) map;
    // keep executing until a reduction over the stable map lands.
    for (let i = 0; i < 100 && Math.abs(pass.averageFLIP - sum / n) > 1e-5; i++) {
        pass.execute(ctx, rd);
        await new Promise((r) => setTimeout(r, 10));
    }
    expectEq(Math.abs(pass.averageFLIP - sum / n) < 1e-5, true, `averageFLIP ${pass.averageFLIP} == ${sum / n}`);
    expectEq(Math.abs(pass.minFLIP - mn) < 1e-6, true, `minFLIP ${pass.minFLIP} == ${mn}`);
    expectEq(Math.abs(pass.maxFLIP - mx) < 1e-6, true, `maxFLIP ${pass.maxFLIP} == ${mx}`);

    console.error(
        `# flip-hdr: start=${p.startExposure.toFixed(4)} delta=${p.exposureDelta.toFixed(4)} n=${p.numExposures} ` +
            `avg=${pass.averageFLIP.toExponential(3)} min=${pass.minFLIP.toExponential(2)} max=${pass.maxFLIP.toExponential(2)}`,
    );
});
