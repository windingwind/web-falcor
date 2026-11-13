/**
 * GBufferRaster linearZ vs GBufferRT linearZ. Native Falcor's raster pass stores SV_Position.z * SV_Position.w = clip-space
 * z, its RT pass stores clip w (view depth); with Falcor's [0,1] perspective the two are related exactly by
 * z_clip = f (w - n) / (f - n). WGSL's fragment position.w is 1 / w, so a literal port of z * w stored z_ndc / w
 * (e.g. 0.174 instead of ~5.6). Checks the raster channel against that relation on every pixel both passes hit.
 */

import { RenderGraph, Scene, createPass, float2, float3, float4 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

type V3 = [number, number, number];
function quad(c: V3[], n: V3) {
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    return {
        vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: new float4(1, 0, 0, 1), texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID: 0,
    };
}

gpuTest("GBufferRaster.linearZMatchesGBufferRT", async ({ device }) => {
    const s = 30;
    const meshes = [
        quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0]),
        quad([[-1, 0, 0.5], [1, 0, 0.5], [1, 1.5, 0.5], [-1, 1.5, 0.5]], [0, 0, 1]),
        quad([[0.5, 0, -3], [2.5, 0, -3], [2.5, 2.5, -3], [0.5, 2.5, -3]], [0, 0, 1]),
    ];
    const scene = new Scene(device, meshes, [{ basic: { baseColor: new float4(0.5, 0.5, 0.5, 1) } }]);
    scene.camera.setPosition(new float3(0.4, 1.6, 6));
    scene.camera.setTarget(new float3(0, 0.6, 0));
    scene.camera.setAspectRatio(1);
    const size = 128;
    const ctx = device.renderContext;
    const render = async (type: string) => {
        const g = new RenderGraph(device, type);
        g.addPass(createPass(device, type, { samplePattern: "Center" }), "P");
        g.markOutput("P.linearZ");
        g.markOutput("P.mask");
        g.onResize(size, size);
        g.setScene(scene);
        g.execute(ctx);
        return {
            z: new Float32Array((await ctx.readTextureSubresource(g.getOutput("P.linearZ")!)).buffer),
            mask: new Float32Array((await ctx.readTextureSubresource(g.getOutput("P.mask")!)).buffer),
        };
    };
    const ras = await render("GBufferRaster");
    const rt = await render("GBufferRT");
    const n = scene.camera.getNearPlane(), f = scene.camera.getFarPlane();
    let hits = 0, bad = 0, maxRel = 0, maxAbsW = 0, maxAt = -1;
    for (let i = 0; i < size * size; i++) {
        if (ras.mask[i] !== 1 || rt.mask[i] !== 1) continue;
        const w = rt.z[i * 2]!;
        const expected = (f * (w - n)) / (f - n);
        const got = ras.z[i * 2]!;
        const rel = Math.abs(got - expected) / expected;
        // silhouette pixels can hit different surfaces in raster vs RT; count them, don't fail on a few
        if (rel > 1e-3) bad++;
        else if (rel > maxRel) { maxRel = rel; maxAt = i; }
        maxAbsW = Math.max(maxAbsW, Math.abs(got - w));
        hits++;
    }
    const c = (size / 2) * size + size / 2;
    console.error(`# GBufferRaster linearZ vs GBufferRT: ${hits} px, max rel err ${maxRel.toExponential(2)} (at ${maxAt}), ${bad} px > 1e-3 (silhouettes); |raster - RT w| <= ${maxAbsW.toFixed(4)} (near ${n}); centre raster ${ras.z[c * 2]} vs RT ${rt.z[c * 2]}`);
    expectEq(hits > size * size * 0.5, true, "enough hit pixels");
    expectEq(maxRel < 1e-4, true, `linearZ rel err ${maxRel}`);
    expectEq(bad <= hits * 0.02, true, `linearZ mismatches ${bad} of ${hits}`);
});
