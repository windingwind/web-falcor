/**
 * Profiler: nested events around the graph (RenderGraphExe::execute() and one
 * child per pass) with GPU times from pass timestampWrites; the parent's GPU
 * time is the sum of its children (every pass is attributed to all active
 * events), CPU times are finite, and the legacy per-pass map still works.
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

        // Frames until GPU times have landed for the graph event.
        let root = profiler.findEvent("/RenderGraphExe::execute()");
        for (let i = 0; i < 40 && !(root && root.gpuTime > 0); i++) {
            graph!.execute(ctx);
            ctx.submit();
            await new Promise((r) => setTimeout(r, 30));
            root = profiler.findEvent("/RenderGraphExe::execute()");
        }
        const events = profiler.getEvents();
        console.error(`# profiler: ${events.map((e) => `${e.name} cpu=${e.cpuTime.toFixed(3)} gpu=${e.gpuTime.toFixed(3)}`).join(" | ")}`);
        expectEq(root !== undefined && root.gpuTime > 0, true, "graph event has GPU time");
        expectEq(events[0]?.name, "/RenderGraphExe::execute()", "graph event first (tree order)");
        const children = events.filter((e) => e.level === 1);
        expectEq(children.length >= 2, true, `one child per pass (${children.length})`);
        expectEq(children.some((e) => e.shortName === "ToneMapping"), true, "ToneMapping event present");
        const childSum = children.reduce((s, e) => s + e.gpuTime, 0);
        expectEq(Math.abs(childSum - root!.gpuTime) < 1e-6, true, `parent GPU time (${root!.gpuTime}) == sum of children (${childSum})`);
        expectEq(events.every((e) => Number.isFinite(e.cpuTime) && e.cpuTime >= 0 && e.cpuTimeAverage >= 0), true, "CPU times finite with EMA");
        expectEq(root!.computeGpuTimeStats().max >= root!.gpuTime * 0.999, true, "history stats cover the landed frames");

        // Legacy per-pass map used by the viewer status line.
        const stats = profiler.getStats();
        expectEq(stats.has("ToneMapping"), true, "ToneMapping labeled in getStats");
        let total = 0;
        for (const v of stats.values()) total += v;
        expectEq(total > 0 && total < 1000, true, `total GPU time ${total}ms sane`);

        // Capture two frames and check the JSON lanes.
        profiler.startCapture();
        for (let i = 0; i < 6; i++) {
            graph!.execute(ctx);
            ctx.submit();
            await new Promise((r) => setTimeout(r, 30));
        }
        const capture = profiler.endCapture()!;
        expectEq(capture.frameCount >= 2, true, `captured frames (${capture.frameCount})`);
        expectEq(capture.lanes.some((l) => l.name === "/RenderGraphExe::execute()/ToneMapping/gpu_time"), true, "gpu_time lane per event");

        // Compute passes (ComputePass.execute) must carry timestamps too.
        const [cg] = await runGraphScript(device, "from falcor import *\ng = RenderGraph('C')\ng.addPass(createPass('ImageLoader', {'filename': 'test_images/smoke_puff.png'}), 'Img')\ng.addPass(createPass('AccumulatePass', {}), 'Acc')\ng.addEdge('Img.dst', 'Acc.input')\ng.markOutput('Acc.output')\nm.addGraph(g)\n");
        await cg!.init();
        cg!.onResize(256, 256);
        let acc = profiler.findEvent("/RenderGraphExe::execute()/Acc");
        for (let i = 0; i < 40 && !(acc && acc.gpuTime > 0); i++) {
            cg!.execute(ctx);
            ctx.submit();
            await new Promise((r) => setTimeout(r, 30));
            acc = profiler.findEvent("/RenderGraphExe::execute()/Acc");
        }
        expectEq(acc !== undefined && acc.gpuTime > 0, true, `compute pass has GPU time (${acc?.gpuTime})`);
    } finally {
        device.profilerHook = null;
    }
});
