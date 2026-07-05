/**
 * M1 pipeline-state GPU tests: Vao/Fbo/GraphicsStateObject/RasterizerState/
 * DepthStencilState/BlendState through the raw draw path.
 */

import {
    BlendFunc,
    BlendOp,
    BlendState,
    BlendStateDesc,
    ComparisonFunc,
    ComputeStateObject,
    DepthStencilState,
    DepthStencilStateDesc,
    Fbo,
    GraphicsStateObject,
    ResourceFormat,
    Topology,
    Vao,
    VertexBufferLayout,
    VertexLayout,
    ResourceBindFlags,
    MemoryType,
} from "@web-falcor/falcor";
import { gpuTest, expectEq, expectArrayEq } from "../harness/registry.js";

const kTriangleWgsl = /* wgsl */ `
struct VSIn { @location(0) pos: vec2f, @location(1) color: vec3f };
struct VSOut { @builtin(position) pos: vec4f, @location(0) color: vec3f };
@vertex fn vsMain(in: VSIn) -> VSOut {
    var out: VSOut;
    out.pos = vec4f(in.pos, 0.5, 1.0);
    out.color = in.color;
    return out;
}
@fragment fn psMain(in: VSOut) -> @location(0) vec4f { return vec4f(in.color, 1.0); }
`;

function makeTriangleVao(device: Parameters<Parameters<typeof gpuTest>[1]>[0]["device"]): Vao {
    // Oversized viewport-covering triangle, clockwise winding (Falcor's default
    // rasterizer state culls back faces with CW = front, D3D convention).
    // Interleaved pos.xy + color.rgb.
    const verts = new Float32Array([
        -3, -3, 1, 0, 0,
        0, 3, 1, 0, 0,
        3, -3, 1, 0, 0,
    ]);
    const vb = device.createBuffer(verts.byteLength, ResourceBindFlags.Vertex, MemoryType.DeviceLocal, verts);
    const layout = new VertexLayout();
    const bufLayout = new VertexBufferLayout();
    bufLayout.addElement("POSITION", 0, ResourceFormat.RG32Float, 1, 0);
    bufLayout.addElement("COLOR", 8, ResourceFormat.RGB32Float, 1, 1);
    bufLayout.stride = 20;
    layout.addBufferLayout(0, bufLayout);
    return new Vao(Topology.TriangleList, layout, [vb]);
}

gpuTest("GraphicsStateObject.drawTriangle", async ({ device }) => {
    const fbo = Fbo.create2D(device, 8, 8, ResourceFormat.RGBA8Unorm);
    const module = device.gpuDevice.createShaderModule({ code: kTriangleWgsl });
    const vao = makeTriangleVao(device);
    const gso = new GraphicsStateObject(device, {
        vertexModule: module,
        vertexEntryPoint: "vsMain",
        fragmentModule: module,
        fragmentEntryPoint: "psMain",
        vertexLayout: vao.vertexLayout,
        colorFormats: fbo.getGpuColorFormats(),
        topology: Topology.TriangleList,
    });
    const ctx = device.renderContext;
    ctx.clearTexture(fbo.getColorTexture(0)!, [0, 0, 0, 1]);
    ctx.drawRaw(gso, vao, fbo, [], 3);
    const px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    // Center pixel fully red.
    const center = (4 * 8 + 4) * 4;
    expectArrayEq([px[center], px[center + 1], px[center + 2]], [255, 0, 0], "center pixel");
});

gpuTest("DepthStencilState.depthTest", async ({ device }) => {
    const fbo = Fbo.create2D(device, 4, 4, ResourceFormat.RGBA8Unorm, ResourceFormat.D32Float);
    const module = device.gpuDevice.createShaderModule({ code: kTriangleWgsl });
    const vao = makeTriangleVao(device);
    const dssAlways = DepthStencilState.create(new DepthStencilStateDesc().setDepthFunc(ComparisonFunc.Always));
    const dssNever = DepthStencilState.create(new DepthStencilStateDesc().setDepthFunc(ComparisonFunc.Never));
    const make = (dss: DepthStencilState) =>
        new GraphicsStateObject(device, {
            vertexModule: module,
            vertexEntryPoint: "vsMain",
            fragmentModule: module,
            fragmentEntryPoint: "psMain",
            vertexLayout: vao.vertexLayout,
            colorFormats: fbo.getGpuColorFormats(),
            depthFormat: fbo.getGpuDepthFormat(),
            depthStencilState: dss,
        });
    const ctx = device.renderContext;
    ctx.clearTexture(fbo.getColorTexture(0)!, [0, 0, 0, 1]);
    ctx.clearDsv(fbo.getDepthStencilTexture()!.getDSV(), 1, 0);
    ctx.drawRaw(make(dssNever), vao, fbo, [], 3);
    let px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    expectEq(px[0], 0, "depth Never: nothing drawn");
    ctx.drawRaw(make(dssAlways), vao, fbo, [], 3);
    px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    expectEq(px[0], 255, "depth Always: drawn");
});

gpuTest("BlendState.additiveBlend", async ({ device }) => {
    const fbo = Fbo.create2D(device, 4, 4, ResourceFormat.RGBA8Unorm);
    const module = device.gpuDevice.createShaderModule({ code: kTriangleWgsl });
    const vao = makeTriangleVao(device);
    const blendDesc = new BlendStateDesc();
    blendDesc.setRtBlend(0, true).setRtParams(0, BlendOp.Add, BlendOp.Add, BlendFunc.One, BlendFunc.One, BlendFunc.One, BlendFunc.One);
    const gso = new GraphicsStateObject(device, {
        vertexModule: module,
        vertexEntryPoint: "vsMain",
        fragmentModule: module,
        fragmentEntryPoint: "psMain",
        vertexLayout: vao.vertexLayout,
        colorFormats: fbo.getGpuColorFormats(),
        blendState: BlendState.create(blendDesc),
    });
    const ctx = device.renderContext;
    ctx.clearTexture(fbo.getColorTexture(0)!, [0, 0, 0.5, 1]);
    ctx.drawRaw(gso, vao, fbo, [], 3);
    const px = await ctx.readTextureSubresource(fbo.getColorTexture(0)!);
    expectEq(px[0], 255, "red added");
    expectEq(px[2]! >= 127 && px[2]! <= 128, true, "blue preserved by additive blend");
});

gpuTest("ComputeStateObject.dispatch", async ({ device }) => {
    const wgsl = `
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        @compute @workgroup_size(32) fn main(@builtin(global_invocation_id) gid: vec3u) {
            data[gid.x] = f32(gid.x) * 0.5;
        }`;
    const cso = new ComputeStateObject(device, {
        module: device.gpuDevice.createShaderModule({ code: wgsl }),
        entryPoint: "main",
    });
    const buffer = device.createBuffer(32 * 4);
    const bindGroup = device.gpuDevice.createBindGroup({
        layout: cso.gpuPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: buffer.gpuBuffer } }],
    });
    device.renderContext.dispatchRaw(cso.gpuPipeline, [bindGroup], 1);
    const result = new Float32Array((await buffer.getBlob()).buffer);
    expectEq(result[10], 5, "computed value");
    buffer.destroy();
});

gpuTest("Fbo.mrtFormatsAndSize", ({ device }) => {
    const fbo = new Fbo();
    const c0 = device.createTexture2D(64, 32, ResourceFormat.RGBA16Float, 1, 1, undefined, ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
    const c2 = device.createTexture2D(64, 32, ResourceFormat.R32Float, 1, 1, undefined, ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
    fbo.attachColorTarget(c0, 0).attachColorTarget(c2, 2);
    expectEq(fbo.width, 64, "fbo width");
    expectEq(fbo.height, 32, "fbo height");
    expectArrayEq(
        fbo.getGpuColorFormats().map((f) => (f === null ? 0 : 1)),
        [1, 0, 1],
        "MRT slots",
    );
});
