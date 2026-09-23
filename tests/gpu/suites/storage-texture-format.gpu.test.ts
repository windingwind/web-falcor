/**
 * Native UAVs take their resource's format; WGSL storage textures declare one, which Slang
 * infers from the element type (float4 -> rgba32float, uint4 -> rgba32uint). ComputePass
 * retargets the declaration to the bound texture, so unmodified shaders write any
 * storage-capable format of the same texel type.
 */

import { ComputePass, ResourceBindFlags, ResourceFormat } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const kShader = `
RWTexture2D<float4> gFloat;
RWTexture2D<uint4> gUint;
[numthreads(4, 4, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    gFloat[id.xy] = float4(id.x * 0.25f, id.y * 0.5f, 0.75f, 1.f);
    gUint[id.xy] = uint4(id.x + 10 * id.y, 0, 0, 0);
}
`;

gpuTest("StorageTextureFormat.retargetsToTheBoundTexture", async ({ device }) => {
    const pass = ComputePass.create(device, { modules: [{ sources: [{ string: kShader, path: "Tests/StorageTextureFormat.cs.slang" }] }], csEntry: "main" });
    const flags = ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource;
    const f16 = device.createTexture2D(4, 4, ResourceFormat.RGBA16Float, 1, 1, undefined, flags);
    const u32 = device.createTexture2D(4, 4, ResourceFormat.R32Uint, 1, 1, undefined, flags);
    pass.getRootVar()["gFloat"] = f16;
    pass.getRootVar()["gUint"] = u32;
    device.gpuDevice.pushErrorScope("validation");
    pass.execute(device.renderContext, 4, 4);
    const error = await device.gpuDevice.popErrorScope();
    expectEq(error?.message ?? "", "", "WebGPU validation");
    const half = new Uint16Array((await device.renderContext.readTextureSubresource(f16, 0)).buffer);
    const uint = new Uint32Array((await device.renderContext.readTextureSubresource(u32, 0)).buffer);
    // Pixel (3, 1): float4(0.75, 0.5, 0.75, 1) -> halves 0x3a00 0x3800 0x3a00 0x3c00; uint 3 + 10.
    const i = 1 * 4 + 3;
    expectEq(Array.from(half.subarray(i * 4, i * 4 + 4)).map((h) => h.toString(16)).join(), "3a00,3800,3a00,3c00", "rgba16float texel");
    expectEq(uint[i], 13, "r32uint texel");
});
