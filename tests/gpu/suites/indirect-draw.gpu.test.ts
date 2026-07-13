/**
 * RasterPass.drawIndirect / drawIndexedIndirect: GPU-driven draw arguments.
 * A fullscreen triangle drawn via an indirect args buffer fills the target;
 * a zero-count command at the next stride draws nothing.
 */

import { Buffer, CullMode, Fbo, MemoryType, RasterPass, RasterizerState, RasterizerStateDesc, ResourceBindFlags, ResourceFormat, ResourceType, Texture, Topology, Vao } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 32;

gpuTest("IndirectDraw.fullscreenTriangle", async ({ device }) => {
    const ctx = device.renderContext;
    const pass = RasterPass.create(device, { path: "WebFalcor/IndirectTest.3d.slang" });
    pass.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
    const target = new Texture(device, {
        type: ResourceType.Texture2D,
        width: size,
        height: size,
        format: ResourceFormat.RGBA8Unorm,
        bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
        mipLevels: 1,
        name: "IndirectDraw::target",
    });
    const fbo = new Fbo();
    fbo.attachColorTarget(target, 0);

    // Two commands: [vertexCount, instanceCount, firstVertex, firstInstance].
    const args = new Buffer(device, {
        size: 8 * 4,
        structSize: 4,
        bindFlags: ResourceBindFlags.IndirectArg,
        memoryType: MemoryType.DeviceLocal,
        name: "IndirectDraw::args",
    });
    args.setBlob(new Uint32Array([3, 1, 0, 0, 0, 1, 0, 0]));

    const redCount = async () => {
        const data = await ctx.readTextureSubresource(target);
        let n = 0;
        for (let i = 0; i < size * size; i++) if (data[i * 4]! === 255) n++;
        return n;
    };

    ctx.clearTexture(target, [0, 0, 0, 0]);
    pass.drawIndirect(ctx, fbo, 1, args, 0);
    expectEq(await redCount(), size * size, "command 0 fills the target");

    ctx.clearTexture(target, [0, 0, 0, 0]);
    pass.drawIndirect(ctx, fbo, 1, args, 16);
    expectEq(await redCount(), 0, "zero-count command draws nothing");

    // Indexed variant: [indexCount, instanceCount, firstIndex, baseVertex, firstInstance].
    const indexBuffer = new Buffer(device, {
        size: 3 * 4,
        structSize: 4,
        bindFlags: ResourceBindFlags.Index,
        memoryType: MemoryType.DeviceLocal,
        name: "IndirectDraw::indices",
    });
    indexBuffer.setBlob(new Uint32Array([0, 1, 2]));
    pass.state.setVao(new Vao(Topology.TriangleList, null, [], indexBuffer));

    const indexedArgs = new Buffer(device, {
        size: 5 * 4,
        structSize: 4,
        bindFlags: ResourceBindFlags.IndirectArg,
        memoryType: MemoryType.DeviceLocal,
        name: "IndirectDraw::indexedArgs",
    });
    indexedArgs.setBlob(new Uint32Array([3, 1, 0, 0, 0]));

    ctx.clearTexture(target, [0, 0, 0, 0]);
    pass.drawIndexedIndirect(ctx, fbo, 1, indexedArgs, 0);
    expectEq(await redCount(), size * size, "indexed indirect fills the target");
});
