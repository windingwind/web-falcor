/**
 * Mirrors Falcor/Utils/Image/ImageProcessing: copyColorChannel writes one source
 * channel to every channel of the destination. §9: native writes through a UAV;
 * most of those formats aren't storage-capable in WebGPU, so a fullscreen
 * fragment pass writes the render target instead (any renderable format).
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import type { Texture } from "../../Core/API/Texture.js";
import { FormatType, getFormatType } from "../../Core/API/Formats.js";
import { ResourceType } from "../../Core/API/Types.js";
import { RuntimeError } from "../../Core/Error.js";

/** Mirrors TextureChannelFlags (single-channel values select the source channel). */
export enum TextureChannelFlags {
    None = 0,
    Red = 1,
    Green = 2,
    Blue = 4,
    Alpha = 8,
    RGB = 7,
    RGBA = 15,
}

type Kind = "f32" | "u32" | "i32";
const kindOf = (t: Texture): Kind => {
    const type = getFormatType(t.format);
    return type === FormatType.Uint ? "u32" : type === FormatType.Sint ? "i32" : "f32";
};

function shader(kind: Kind): string {
    return /* wgsl */ `
@group(0) @binding(0) var gSrc: texture_2d<${kind}>;
@group(0) @binding(1) var<uniform> gChannel: vec4u;

@vertex fn vsMain(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((vid << 1u) & 2u), f32(vid & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn psMain(@builtin(position) pos: vec4f) -> @location(0) vec4<${kind}> {
    let v = textureLoad(gSrc, vec2<i32>(pos.xy), i32(gChannel.y))[gChannel.x];
    return vec4<${kind}>(v);
}
`;
}

export class ImageProcessing {
    private pipelines = new Map<string, GPURenderPipeline>();

    constructor(private readonly device: Device) {}

    /** Mirrors copyColorChannel(ctx, srcSRV, dstUAV, srcMask) for mip `srcMip` -> `dstMip`. */
    copyColorChannel(ctx: RenderContext, src: Texture, dst: Texture, srcMask: TextureChannelFlags, srcMip = 0, dstMip = 0): void {
        if (src.type !== ResourceType.Texture2D) throw new RuntimeError("Source resource type must be Texture2D");
        if (dst.type !== ResourceType.Texture2D) throw new RuntimeError("Source resource type must be Texture2D");
        const w = Math.max(1, src.width >> srcMip);
        const h = Math.max(1, src.height >> srcMip);
        if (w !== Math.max(1, dst.width >> dstMip) || h !== Math.max(1, dst.height >> dstMip)) throw new RuntimeError("Source and destination views must have matching dimensions");
        const kind = kindOf(src);
        if ((kind === "f32") !== (kindOf(dst) === "f32")) throw new RuntimeError("Source and destination texture must have matching format type");
        const channelIndex = { [TextureChannelFlags.Red]: 0, [TextureChannelFlags.Green]: 1, [TextureChannelFlags.Blue]: 2, [TextureChannelFlags.Alpha]: 3 }[srcMask as number];
        if (channelIndex === undefined) throw new RuntimeError("'channelMask' parameter must be a single color channel.");

        const dstKind = kindOf(dst);
        const key = `${dst.gpuFormat}|${kind}`;
        let pipeline = this.pipelines.get(key);
        if (!pipeline) {
            const gpu = this.device.gpuDevice;
            // Source and destination integer kinds may differ in signedness; the shader emits the destination's.
            const code = shader(kind).replace(`-> @location(0) vec4<${kind}>`, `-> @location(0) vec4<${dstKind}>`).replace(`return vec4<${kind}>(v);`, `return vec4<${dstKind}>(v);`);
            const module = gpu.createShaderModule({ code });
            pipeline = gpu.createRenderPipeline({
                layout: "auto",
                vertex: { module, entryPoint: "vsMain" },
                fragment: { module, entryPoint: "psMain", targets: [{ format: dst.gpuFormat }] },
                primitive: { topology: "triangle-list" },
            });
            this.pipelines.set(key, pipeline);
        }
        // Per-call constants: queue writes would run ahead of passes already recorded.
        const uniforms = this.device.gpuDevice.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Uint32Array(uniforms.getMappedRange()).set([channelIndex, srcMip, 0, 0]);
        uniforms.unmap();
        const bindGroup = this.device.gpuDevice.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: src.getSRV() },
                { binding: 1, resource: { buffer: uniforms } },
            ],
        });
        const pass = ctx.getEncoder().beginRenderPass({
            colorAttachments: [{ view: dst.getRTV(dstMip), loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }],
        });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3);
        pass.end();
    }
}
