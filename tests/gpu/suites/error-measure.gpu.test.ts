/**
 * ErrorMeasurePass: per-pixel difference + GPU-reduced mean error. Case 1
 * verifies the math on constant images (|0.75-0.5|^2 = 0.0625); case 2 loads
 * the reference from an EXR file and self-diffs to ~0.
 */

import { Properties, RenderData, ResourceBindFlags, ResourceFormat, decodeExr } from "@web-falcor/falcor";
import { ErrorMeasurePass } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

async function waitForMeasurement(pass: ErrorMeasurePass, run: () => void): Promise<void> {
    for (let i = 0; i < 10 && !pass.measurements.valid; i++) {
        run();
        await new Promise((r) => setTimeout(r, 50));
    }
}

gpuTest("ErrorMeasure.constantDifference", async ({ device }) => {
    const size = 8;
    const mk = (v: number) => device.createTexture2D(size, size, ResourceFormat.RGBA32Float, 1, 1, new Float32Array(size * size * 4).fill(v));
    const src = mk(0.75);
    const ref = mk(0.5);
    const out = device.createTexture2D(size, size, ResourceFormat.RGBA32Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);

    const pass = new ErrorMeasurePass(device, new Properties({ SelectedOutputId: "Difference" }));
    const rd = new RenderData(new Map([["Source", src], ["Reference", ref], ["Output", out]]), [size, size]);
    await waitForMeasurement(pass, () => pass.execute(device.renderContext, rd));

    expectEq(pass.measurements.valid, true, "measurements arrived");
    expectEq(Math.abs(pass.measurements.avgError - 0.0625) < 1e-6, true, `avgError ${pass.measurements.avgError} == 0.0625`);
    const outPx = new Float32Array((await device.renderContext.readTextureSubresource(out)).buffer);
    expectEq(Math.abs(outPx[0]! - 0.0625) < 1e-6, true, `difference output ${outPx[0]}`);
});

gpuTest("ErrorMeasure.exrFileReferenceSelfDiff", async ({ device }) => {
    const path = "../../tests/oracle/out-native/oracle-usd.ToneMapper.dst.0.exr";
    const exr = decodeExr(await (await fetch("/tests/oracle/out-native/oracle-usd.ToneMapper.dst.0.exr")).arrayBuffer());
    // Perturb a quarter of the pixels so the expected error is nonzero
    // (a pure self-diff cannot distinguish broken-zero from true-zero).
    const perturbed = exr.data.slice();
    const quarter = (exr.width * exr.height) / 4;
    for (let i = 0; i < quarter; i++) perturbed[i * 4] = perturbed[i * 4]! + 0.5;
    const src = device.createTexture2D(exr.width, exr.height, ResourceFormat.RGBA32Float, 1, 1, perturbed);
    const out = device.createTexture2D(exr.width, exr.height, ResourceFormat.RGBA32Float, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);

    const pass = new ErrorMeasurePass(device, new Properties({ ReferenceImagePath: path }));
    await pass.initAsync();
    const rd = new RenderData(new Map([["Source", src], ["Output", out]]), [exr.width, exr.height]);
    await waitForMeasurement(pass, () => pass.execute(device.renderContext, rd));

    expectEq(pass.measurements.valid, true, "measurements arrived");
    // diffSqr 0.25 on R over a quarter of pixels -> avg (0.0625 + 0 + 0)/3.
    expectEq(Math.abs(pass.measurements.avgError - 0.0625 / 3) < 1e-4, true, `perturbed avgError ${pass.measurements.avgError}`);
});
