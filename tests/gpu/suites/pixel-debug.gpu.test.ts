/**
 * PixelDebug end-to-end: a compute kernel prints uint/float/int/bool/vector
 * values via the portable override; the host captures records for the
 * selected pixel only, in order, with exact bit-decoded values, plus the
 * assert record from exactly one failing pixel.
 */

import { ComputePass, PixelDebug, PrintValueType } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("PixelDebug.capturesSelectedPixelPrints", async ({ device }) => {
    const debug = new PixelDebug(device);
    debug.enabled = true;
    debug.selectedPixel = [3, 5];

    const pass = ComputePass.create(device, { path: "WebFalcor/PixelDebugTest.cs.slang", defines: PixelDebug.getDefines() });
    const ctx = device.renderContext;
    debug.beginFrame(ctx);
    debug.prepareProgram(pass.getRootVar());
    pass.execute(ctx, 16, 16);
    debug.endFrame();
    ctx.submit();

    let prints = debug.getPrintRecords();
    for (let i = 0; i < 50 && prints.length === 0; i++) {
        await new Promise((r) => setTimeout(r, 50));
        prints = debug.getPrintRecords();
    }

    console.error(`# pixel-debug: ${prints.length} prints, ${debug.getAssertRecords().length} asserts`);
    for (const p of prints) console.error(`#   type=${PrintValueType[p.type]} values=${p.values.join(",")}`);

    expectEq(prints.length, 5, "one record per print call, selected pixel only");
    expectEq(prints[0]!.type, PrintValueType.Uint, "uint type");
    expectEq(prints[0]!.values[0], 3, "pixel.x");
    expectEq(prints[1]!.type, PrintValueType.Float, "float type");
    expectEq(prints[1]!.values[0], 1.5, "x * 0.5");
    expectEq(prints[2]!.type, PrintValueType.Int, "int type");
    expectEq(prints[2]!.values[0], -6, "-y - 1");
    expectEq(prints[3]!.type, PrintValueType.Bool, "bool type");
    expectEq(prints[3]!.values[0], false, "x=3 is odd");
    expectEq(prints[4]!.values.join(","), "1.5,2.5,3.5", "float3 components");

    const asserts = debug.getAssertRecords();
    expectEq(asserts.length, 1, "exactly one failing pixel");
    expectEq(asserts[0]!.launchX, 3, "assert launch x");
    expectEq(asserts[0]!.launchY, 5, "assert launch y");
});
