/**
 * Per-pass GPU profiler: timestampWrites on every GPU pass, labeled by the
 * RenderGraph, resolved per frame. Verifies labels + sane timings over the
 * upstream ToneMapping graph.
 */

import { Profiler, initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("Profiler.perPassTimings", async ({ device }) => {
    if (!device.hasFeature("timestamp-query")) {
        console.error("# profiler: timestamp-query unavailable, skipping");
        return;
    }
    const profiler = new Profiler(device);
    device.enableProfiler(profiler);
    try {
        await initScripting("/node_modules/pyodide");
        const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py")).text();
        const [graph] = await runGraphScript(device, source);
        await graph!.init();
        graph!.onResize(256, 256);
        const ctx = device.renderContext;

        let stats = new Map<string, number>();
        for (let i = 0; i < 20 && stats.size === 0; i++) {
            graph!.execute(ctx);
            ctx.submit();
            await new Promise((r) => setTimeout(r, 30));
            stats = profiler.getStats();
        }

        console.error(`# profiler: ${[...stats].map(([k, v]) => `${k}=${v.toFixed(3)}ms`).join(" ")}`);
        expectEq(stats.size >= 2, true, `labels present (${stats.size})`);
        expectEq(stats.has("ToneMapping"), true, "ToneMapping labeled");
        let total = 0;
        let finite = true;
        for (const v of stats.values()) {
            total += v;
            finite &&= Number.isFinite(v) && v >= 0 && v < 1000;
        }
        expectEq(finite, true, "timings finite and sane");
        expectEq(total > 0, true, `total GPU time ${total}ms > 0`);
    } finally {
        device.profilerHook = null;
    }
});
