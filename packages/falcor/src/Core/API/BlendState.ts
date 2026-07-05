/**
 * Blend state mirroring Falcor/Core/API/BlendState.h.
 * Src1* funcs require WebGPU 'dual-source-blending'; validated at pipeline creation.
 */

export enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

export enum BlendFunc {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    BlendFactor,
    OneMinusBlendFactor,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

export interface RtBlendDesc {
    blendEnabled: boolean;
    rgbBlendOp: BlendOp;
    alphaBlendOp: BlendOp;
    srcRgbFunc: BlendFunc;
    dstRgbFunc: BlendFunc;
    srcAlphaFunc: BlendFunc;
    dstAlphaFunc: BlendFunc;
    writeMask: { r: boolean; g: boolean; b: boolean; a: boolean };
}

export function defaultRtBlendDesc(): RtBlendDesc {
    return {
        blendEnabled: false,
        rgbBlendOp: BlendOp.Add,
        alphaBlendOp: BlendOp.Add,
        srcRgbFunc: BlendFunc.One,
        dstRgbFunc: BlendFunc.Zero,
        srcAlphaFunc: BlendFunc.One,
        dstAlphaFunc: BlendFunc.Zero,
        writeMask: { r: true, g: true, b: true, a: true },
    };
}

export class BlendStateDesc {
    blendFactor: [number, number, number, number] = [0, 0, 0, 0];
    independentBlend = false;
    alphaToCoverage = false;
    rtDescs: RtBlendDesc[] = [defaultRtBlendDesc()];

    setBlendFactor(factor: [number, number, number, number]): this {
        this.blendFactor = factor;
        return this;
    }
    setIndependentBlend(enabled: boolean): this {
        this.independentBlend = enabled;
        return this;
    }
    setAlphaToCoverage(enabled: boolean): this {
        this.alphaToCoverage = enabled;
        return this;
    }
    setRtBlend(rtIndex: number, enabled: boolean): this {
        while (this.rtDescs.length <= rtIndex) this.rtDescs.push(defaultRtBlendDesc());
        this.rtDescs[rtIndex]!.blendEnabled = enabled;
        return this;
    }
    setRtParams(
        rtIndex: number,
        rgbOp: BlendOp,
        alphaOp: BlendOp,
        srcRgbFunc: BlendFunc,
        dstRgbFunc: BlendFunc,
        srcAlphaFunc: BlendFunc,
        dstAlphaFunc: BlendFunc,
    ): this {
        while (this.rtDescs.length <= rtIndex) this.rtDescs.push(defaultRtBlendDesc());
        Object.assign(this.rtDescs[rtIndex]!, { rgbBlendOp: rgbOp, alphaBlendOp: alphaOp, srcRgbFunc, dstRgbFunc, srcAlphaFunc, dstAlphaFunc });
        return this;
    }
    setRenderTargetWriteMask(rtIndex: number, r: boolean, g: boolean, b: boolean, a: boolean): this {
        while (this.rtDescs.length <= rtIndex) this.rtDescs.push(defaultRtBlendDesc());
        this.rtDescs[rtIndex]!.writeMask = { r, g, b, a };
        return this;
    }
}

const kBlendOpMap: Record<BlendOp, GPUBlendOperation> = {
    [BlendOp.Add]: "add",
    [BlendOp.Subtract]: "reverse-subtract", // Falcor: subtract src1 FROM src2 (dst - src)
    [BlendOp.ReverseSubtract]: "subtract",
    [BlendOp.Min]: "min",
    [BlendOp.Max]: "max",
};

const kBlendFuncMap: Record<BlendFunc, GPUBlendFactor> = {
    [BlendFunc.Zero]: "zero",
    [BlendFunc.One]: "one",
    [BlendFunc.SrcColor]: "src",
    [BlendFunc.OneMinusSrcColor]: "one-minus-src",
    [BlendFunc.DstColor]: "dst",
    [BlendFunc.OneMinusDstColor]: "one-minus-dst",
    [BlendFunc.SrcAlpha]: "src-alpha",
    [BlendFunc.OneMinusSrcAlpha]: "one-minus-src-alpha",
    [BlendFunc.DstAlpha]: "dst-alpha",
    [BlendFunc.OneMinusDstAlpha]: "one-minus-dst-alpha",
    [BlendFunc.BlendFactor]: "constant",
    [BlendFunc.OneMinusBlendFactor]: "one-minus-constant",
    [BlendFunc.SrcAlphaSaturate]: "src-alpha-saturated",
    [BlendFunc.Src1Color]: "src1",
    [BlendFunc.OneMinusSrc1Color]: "one-minus-src1",
    [BlendFunc.Src1Alpha]: "src1-alpha",
    [BlendFunc.OneMinusSrc1Alpha]: "one-minus-src1-alpha",
};

export class BlendState {
    constructor(public readonly desc: BlendStateDesc) {}

    static create(desc: BlendStateDesc = new BlendStateDesc()): BlendState {
        return new BlendState(desc);
    }

    /** Lowered per-target GPUBlendState + write mask for pipeline creation. */
    getGpuTargetState(rtIndex: number): { blend: GPUBlendState | undefined; writeMask: GPUColorWriteFlags } {
        const rt = this.desc.rtDescs[this.desc.independentBlend ? rtIndex : 0] ?? defaultRtBlendDesc();
        const writeMask =
            (rt.writeMask.r ? GPUColorWrite.RED : 0) |
            (rt.writeMask.g ? GPUColorWrite.GREEN : 0) |
            (rt.writeMask.b ? GPUColorWrite.BLUE : 0) |
            (rt.writeMask.a ? GPUColorWrite.ALPHA : 0);
        if (!rt.blendEnabled) return { blend: undefined, writeMask };
        return {
            blend: {
                color: { operation: kBlendOpMap[rt.rgbBlendOp], srcFactor: kBlendFuncMap[rt.srcRgbFunc], dstFactor: kBlendFuncMap[rt.dstRgbFunc] },
                alpha: { operation: kBlendOpMap[rt.alphaBlendOp], srcFactor: kBlendFuncMap[rt.srcAlphaFunc], dstFactor: kBlendFuncMap[rt.dstAlphaFunc] },
            },
            writeMask,
        };
    }
}
