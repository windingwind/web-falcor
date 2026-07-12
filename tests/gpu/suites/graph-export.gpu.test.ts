/**
 * RenderGraph .py export (mirrors RenderGraphExporter) + removeEdge/
 * unmarkOutput: exporting the upstream PathTracer graph, re-importing it,
 * and exporting again must reach a fixpoint (identical script), proving the
 * export captures passes+props+edges+outputs exactly as the importer reads.
 */

import { initScripting, runGraphScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("RenderGraphExport.roundTripFixpoint", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/PathTracer.py")).text();
    const [g1] = await runGraphScript(device, source);

    const script = g1!.exportScript();
    console.error(`# exported ${script.split("\n").length} lines`);
    const [g2] = await runGraphScript(device, script);
    expectEq(g2!.exportScript(), script, "round-trip fixpoint");
    expectEq(g2!.getOutputNames().join(","), g1!.getOutputNames().join(","), "outputs preserved");
    expectEq(
        g2!.getPasses().map((p) => `${p.name}:${p.pass.type}`).join(","),
        g1!.getPasses().map((p) => `${p.name}:${p.pass.type}`).join(","),
        "passes preserved",
    );

    g2!.removeEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    expectEq(g2!.exportScript().includes('"VBufferRT.vbuffer"'), false, "edge removed");
    g2!.unmarkOutput("PathTracer.color");
    expectEq(g2!.getOutputNames().includes("PathTracer.color"), false, "output unmarked");
    expectEq(g2!.getOutputNames().length > 0, true, "other outputs intact");
});
