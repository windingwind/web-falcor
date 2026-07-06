/**
 * ParameterBlock regression: uniform members of a struct that mixes nested
 * uniform structs with resource members (the SceneDebugger CB shape) must
 * land at the WGSL std140 offsets (vec2 after a scalar goes to offset 8,
 * not the HLSL-packed 4).
 */

import { Buffer, ComputePass, MemoryType, ResourceBindFlags, ResourceFormat, ResourceType, Texture } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ParameterBlock.mixedStructUniformOffsets", async ({ device }) => {
    const pass = ComputePass.create(device, { path: "WebFalcor/MixedStructRepro.cs.slang" });
    const root = pass.getRootVar();
    const m = root["CB"]!["gMixed"]!;
    const p = m["params"]!;
    p["mode"] = 7;
    p["frameDim"] = [16, 16];
    p["frameCount"] = 100;

    const data = new Buffer(device, { size: 4, structSize: 4, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "repro::data" });
    data.setBlob(new Uint8Array(new Uint32Array([11]).buffer));
    const result = new Buffer(device, { size: 16, structSize: 4, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "repro::result" });
    const output = new Texture(device, { type: ResourceType.Texture2D, width: 16, height: 16, format: ResourceFormat.RGBA32Float, bindFlags: ResourceBindFlags.UnorderedAccess, name: "repro::out" });
    m["data"] = data;
    m["result"] = result;
    m["output"] = output;

    const ctx = device.renderContext;
    pass.execute(ctx, 16, 16);
    const readback = new Uint32Array((await ctx.readBuffer(result)).buffer);
    console.error(`# mixedStruct: mode=${readback[0]} frameDim=${readback[1]},${readback[2]} fc+data=${readback[3]}`);
    expectEq(readback[0], 7, "mode");
    expectEq(readback[1], 16, "frameDim.x");
    expectEq(readback[2], 16, "frameDim.y");
    expectEq(readback[3], 111, "frameCount + data[0]");
});
