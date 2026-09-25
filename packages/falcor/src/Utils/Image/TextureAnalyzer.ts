/**
 * Mirrors Utils/Image/TextureAnalyzer: per texture, which channels are constant (bits 0-3 of the
 * mask), each channel's numerical range (bits 4-19: pos/neg/inf/NaN per channel), the constant
 * value (the top-left texel) and the min/max values clamped to zero, written as 64-byte Results.
 */

import type { Device } from "../../Core/API/Device.js";
import type { Buffer } from "../../Core/API/Buffer.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Texture as TextureClass, type Texture } from "../../Core/API/Texture.js";
import { FormatType, ResourceFormat, getFormatType } from "../../Core/API/Formats.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { RuntimeError } from "../../Core/Error.js";

const kShaderFilename = "Utils/Image/TextureAnalyzer.cs.slang";

/** Mirrors TextureAnalyzer::Result::RangeFlags. */
export enum TextureAnalyzerRangeFlags {
    Pos = 0x1,
    Neg = 0x2,
    Inf = 0x4,
    NaN = 0x8,
}

/** Mirrors TextureAnalyzer::Result (64 bytes). */
export class TextureAnalyzerResult {
    constructor(
        readonly mask: number,
        readonly value: [number, number, number, number],
        readonly minValue: [number, number, number, number],
        readonly maxValue: [number, number, number, number],
    ) {}

    static fromBytes(bytes: Uint8Array, offset = 0): TextureAnalyzerResult {
        const u = new Uint32Array(bytes.buffer, bytes.byteOffset + offset, 16);
        const f = new Float32Array(bytes.buffer, bytes.byteOffset + offset, 16);
        const v = (o: number) => [f[o]!, f[o + 1]!, f[o + 2]!, f[o + 3]!] as [number, number, number, number];
        return new TextureAnalyzerResult(u[0]!, v(4), v(8), v(12));
    }

    /** True if all channels in `channelMask` (TextureChannelFlags bits) are constant. */
    isConstant(channelMask: number): boolean {
        return (this.mask & channelMask) === 0;
    }
    getRange(channelMask: number): number {
        let range = 0;
        for (let i = 0; i < 4; i++) if (channelMask & (1 << i)) range |= this.mask >>> (4 + 4 * i);
        return range & 0xf;
    }
    isPos(channelMask: number): boolean { return (this.getRange(channelMask) & TextureAnalyzerRangeFlags.Pos) !== 0; }
    isNeg(channelMask: number): boolean { return (this.getRange(channelMask) & TextureAnalyzerRangeFlags.Neg) !== 0; }
    isInf(channelMask: number): boolean { return (this.getRange(channelMask) & TextureAnalyzerRangeFlags.Inf) !== 0; }
    isNaN(channelMask: number): boolean { return (this.getRange(channelMask) & TextureAnalyzerRangeFlags.NaN) !== 0; }
}

export class TextureAnalyzer {
    private readonly clearPass: ComputePass;
    private readonly analyzePass: ComputePass;
    /** The clear pass shares the struct, so its (unread) input binding needs a texture. */
    private readonly dummyInput: Texture;

    constructor(device: Device) {
        this.dummyInput = new TextureClass(device, { type: ResourceType.Texture2D, width: 1, height: 1, format: ResourceFormat.RGBA32Float, bindFlags: ResourceBindFlags.ShaderResource });
        this.clearPass = ComputePass.create(device, { path: kShaderFilename, csEntry: "clear" });
        this.analyzePass = ComputePass.create(device, { path: kShaderFilename, csEntry: "analyze" });
    }

    /** Mirrors TextureAnalyzer::getResultSize. */
    static getResultSize(): number {
        return 64;
    }

    /** Mirrors TextureAnalyzer::analyze for one texture subresource. */
    analyze(ctx: RenderContext, input: Texture, mipLevel: number, arraySlice: number, result: Buffer, resultOffset = 0, clearResult = true): void {
        if (resultOffset + 64 > result.size) throw new RuntimeError("TextureAnalyzer: result buffer too small");
        this.checkFormatSupport(input, mipLevel, arraySlice);
        if (clearResult) this.clear(ctx, result, resultOffset, 1);
        const dim: [number, number] = [Math.max(1, input.width >> mipLevel), Math.max(1, input.height >> mipLevel)];
        const v = this.analyzePass.getRootVar()["gTextureAnalyzer"] as ShaderVar;
        v["input"] = input.getSRV(mipLevel, 1, arraySlice, 1);
        v["result"] = result;
        v["resultOffset"] = resultOffset;
        v["inputDim"] = dim;
        this.analyzePass.execute(ctx, dim[0], dim[1], 1);
    }

    /** Mirrors TextureAnalyzer::analyze for a list of textures (mip 0, slice 0), results packed back to back. */
    analyzeAll(ctx: RenderContext, inputs: Texture[], result: Buffer, clearResult = true): void {
        if (clearResult) this.clear(ctx, result, 0, inputs.length);
        inputs.forEach((t, i) => this.analyze(ctx, t, 0, 0, result, i * 64, false));
    }

    /** Mirrors TextureAnalyzer::clear. */
    clear(ctx: RenderContext, result: Buffer, resultOffset: number, resultCount: number): void {
        const v = this.clearPass.getRootVar()["gTextureAnalyzer"] as ShaderVar;
        v["input"] = this.dummyInput;
        v["result"] = result;
        v["resultOffset"] = resultOffset;
        v["inputDim"] = [resultCount, 1];
        this.clearPass.execute(ctx, resultCount, 1, 1);
    }

    private checkFormatSupport(input: Texture, mipLevel: number, arraySlice: number): void {
        if (input.type === ResourceType.Texture3D && input.depth > 1) throw new RuntimeError("3D textures are not supported");
        if (mipLevel >= input.mipCount || arraySlice >= input.arraySize) throw new RuntimeError("Mip level and/or array slice is out of range");
        if (input.sampleCount !== 1) throw new RuntimeError("Multi-sampled textures are not supported");
        const type = getFormatType(input.format);
        if (type === FormatType.Sint || type === FormatType.Uint) throw new RuntimeError(`Format ${ResourceFormat[input.format]} is not supported`);
    }
}
