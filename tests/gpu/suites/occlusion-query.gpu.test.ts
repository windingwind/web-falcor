/**
 * Occlusion queries (QueryHeap + RasterPass.setOcclusionQuery): a near
 * triangle passes all samples; a far triangle drawn after it is fully
 * depth-rejected and its query reads zero.
 */

import { CullMode, DepthStencilState, DepthStencilStateDesc, Fbo, QueryHeap, QueryHeapType, RasterPass, RasterizerState, RasterizerStateDesc, ResourceBindFlags, ResourceFormat, ResourceType, Texture, type ShaderVar } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 32;

gpuTest("OcclusionQuery.depthRejectedDrawReadsZero", async ({ device }) => {
    const ctx = device.renderContext;
    const pass = RasterPass.create(device, { path: "WebFalcor/OcclusionTest.3d.slang" });
    pass.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
    pass.state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc()));

    const target = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        format: ResourceFormat.RGBA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        mipLevels: 1,
        name: "OcclusionQuery::target",
    });
    const depth = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        format: ResourceFormat.D32Float,
        bindFlags: ResourceBindFlags.DepthStencil,
        mipLevels: 1,
        name: "OcclusionQuery::depth",
    });
    const fbo = new Fbo();
    fbo.attachColorTarget(target, 0);
    fbo.attachDepthStencilTarget(depth);
    ctx.clearTexture(target, [0, 0, 0, 0]);
    ctx.clearDsv(depth.getDSV(), 1, 0);

    const heap = new QueryHeap(device, QueryHeapType.Occlusion, 2);

    (pass.getRootVar()["CB"] as ShaderVar)["gDepth"] = 0.25;
    pass.setOcclusionQuery(heap, 0);
    pass.draw(ctx, fbo, 3);
    ctx.submit(); // flush before the cbuffer changes for the second draw

    (pass.getRootVar()["CB"] as ShaderVar)["gDepth"] = 0.75;
    pass.setOcclusionQuery(heap, 1);
    pass.draw(ctx, fbo, 3);
    pass.setOcclusionQuery(null);

    const results = await heap.resolve(ctx);
    console.error(`# occlusion: near=${results[0]} far=${results[1]}`);
    expectEq(results[0]! > 0n, true, `near draw visible (${results[0]})`);
    expectEq(results[1]!, 0n, "far draw fully occluded");
    heap.destroy();
});
