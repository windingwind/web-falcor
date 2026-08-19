import { GltfImporter, LightType, RenderGraph, createPass, float3 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("RTXDIProbe.quadEmissive", async ({ device }) => {
    const size = 256;
    const scene = await GltfImporter.importFromUrl(device, "/tests/oracle/assets/quad-emissive.gltf", [
        { type: LightType.Point, posW: new float3(0.5, 0.5, 1.5), intensity: new float3(3, 3, 3) },
    ]);
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);
    const graph = new RenderGraph(device, "RTXDIProbe");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "RTXDIPass", {}), "RTXDIPass");
    graph.addEdge("VBufferRT.vbuffer", "RTXDIPass.vbuffer");
    graph.addEdge("VBufferRT.mvec", "RTXDIPass.mvec");
    graph.markOutput("RTXDIPass.color");
    graph.setScene(scene);
    const ctx = device.renderContext;
    for (let f = 0; f < 4; f++) graph.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("RTXDIPass.color")!)).buffer);
    let lit = 0;
    let sum = 0;
    for (let i = 0; i < size * size; i++) {
        const v = web[i * 4]! + web[i * 4 + 1]! + web[i * 4 + 2]!;
        sum += v;
        if (v > 0.01) lit++;
    }
    console.error(`# rtxdi-probe: mean=${(sum / (size * size * 3)).toFixed(4)} litPx=${lit}`);
    expectEq(lit > 1000, true, `standalone RTXDIPass lit px ${lit}`);
});
