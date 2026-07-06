/**
 * M7 (§11.1 .py-first): UNMODIFIED .pyscene files drive web scenes through the
 * Pyodide SceneBuilder bridge, reproducing the exact renders the same files
 * produce natively (compared against the existing native oracles).
 */

import { RenderGraph, createPass, float3, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

async function renderAndCompare(
    device: any,
    pyscenePath: string,
    oracleExr: string,
    label: string,
): Promise<void> {
    const size = 256;
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch(pyscenePath)).text();
    const scene = await runSceneScript(device, source, pyscenePath.split("/").slice(0, -1).join("/"));

    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, `PySceneGraph_${label}`);
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 3 }), "MinimalPathTracer");
    graph.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer");
    graph.markOutput("MinimalPathTracer.color");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("MinimalPathTracer.color")!)).buffer);

    const res = await fetch(oracleExr);
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let badPixels = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const webIdx = (y * size + x) * 4;
            const natIdx = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[webIdx + c]! - data[natIdx + c]!);
                sum += d;
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.05) badPixels++;
        }
    }
    const mean = sum / (size * size * 3);
    console.error(`# pyscene-${label}: meanAbs=${mean.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `radiance mean abs diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels}`);
}

gpuTest("PyScene.importSceneMatchesOracle", async ({ device }) => {
    // importScene(glTF) + PointLight path.
    await renderAndCompare(
        device,
        "/tests/oracle/assets/oracle-pt.pyscene",
        "/tests/oracle/out-native/oracle-pt.MinimalPathTracer.color.0.exr",
        "import",
    );
});

gpuTest("PyScene.builderGeometryMatchesOracle", async ({ device }) => {
    // TriangleMesh.createQuad + ClothMaterial + Transform + addMeshInstance path.
    await renderAndCompare(
        device,
        "/tests/oracle/assets/oracle-pt-cloth.pyscene",
        "/tests/oracle/out-native/oracle-pt-cloth.MinimalPathTracer.color.0.exr",
        "cloth",
    );
});
