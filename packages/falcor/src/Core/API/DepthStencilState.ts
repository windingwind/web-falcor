/**
 * Depth-stencil state mirroring Falcor/Core/API/DepthStencilState.h.
 * WebGPU limitation: one stencil read/write mask shared by front and back faces
 * (D3D12/Vulkan allow per-face); per-face ops are supported.
 */

import { ComparisonFunc, toGpuCompareFunction } from "./Types.js";

export enum Face {
    Front,
    Back,
    FrontAndBack,
}

export enum StencilOp {
    Keep,
    Zero,
    Replace,
    Increase,
    IncreaseSaturate,
    Decrease,
    DecreaseSaturate,
    Invert,
}

export interface StencilDesc {
    func: ComparisonFunc;
    stencilFailOp: StencilOp;
    depthFailOp: StencilOp;
    depthStencilPassOp: StencilOp;
}

const kDefaultStencil: StencilDesc = {
    func: ComparisonFunc.Disabled,
    stencilFailOp: StencilOp.Keep,
    depthFailOp: StencilOp.Keep,
    depthStencilPassOp: StencilOp.Keep,
};

const kStencilOpMap: Record<StencilOp, GPUStencilOperation> = {
    [StencilOp.Keep]: "keep",
    [StencilOp.Zero]: "zero",
    [StencilOp.Replace]: "replace",
    [StencilOp.Increase]: "increment-wrap",
    [StencilOp.IncreaseSaturate]: "increment-clamp",
    [StencilOp.Decrease]: "decrement-wrap",
    [StencilOp.DecreaseSaturate]: "decrement-clamp",
    [StencilOp.Invert]: "invert",
};

export class DepthStencilStateDesc {
    depthEnabled = true;
    writeDepth = true;
    depthFunc = ComparisonFunc.Less;
    stencilEnabled = false;
    stencilReadMask = 0xff;
    stencilWriteMask = 0xff;
    stencilRef = 0;
    stencilFront: StencilDesc = { ...kDefaultStencil };
    stencilBack: StencilDesc = { ...kDefaultStencil };

    setDepthEnabled(enabled: boolean): this { this.depthEnabled = enabled; return this; }
    setDepthWriteMask(write: boolean): this { this.writeDepth = write; return this; }
    setDepthFunc(func: ComparisonFunc): this { this.depthFunc = func; return this; }
    setStencilEnabled(enabled: boolean): this { this.stencilEnabled = enabled; return this; }
    setStencilReadMask(mask: number): this { this.stencilReadMask = mask; return this; }
    setStencilWriteMask(mask: number): this { this.stencilWriteMask = mask; return this; }
    setStencilRef(ref: number): this { this.stencilRef = ref; return this; }
    setStencilFunc(face: Face, func: ComparisonFunc): this {
        if (face !== Face.Back) this.stencilFront.func = func;
        if (face !== Face.Front) this.stencilBack.func = func;
        return this;
    }
    setStencilOp(face: Face, stencilFail: StencilOp, depthFail: StencilOp, depthStencilPass: StencilOp): this {
        const set = (d: StencilDesc) => Object.assign(d, { stencilFailOp: stencilFail, depthFailOp: depthFail, depthStencilPassOp: depthStencilPass });
        if (face !== Face.Back) set(this.stencilFront);
        if (face !== Face.Front) set(this.stencilBack);
        return this;
    }
}

export class DepthStencilState {
    constructor(public readonly desc: DepthStencilStateDesc) {}

    static create(desc: DepthStencilStateDesc = new DepthStencilStateDesc()): DepthStencilState {
        return new DepthStencilState(desc);
    }

    /** Lowers to the depth/stencil portion of a GPUDepthStencilState (format supplied by the FBO). */
    getGpuDepthStencilState(format: GPUTextureFormat): GPUDepthStencilState {
        const d = this.desc;
        const face = (s: StencilDesc): GPUStencilFaceState => ({
            compare: d.stencilEnabled ? (toGpuCompareFunction(s.func) ?? "always") : "always",
            failOp: kStencilOpMap[s.stencilFailOp],
            depthFailOp: kStencilOpMap[s.depthFailOp],
            passOp: kStencilOpMap[s.depthStencilPassOp],
        });
        return {
            format,
            depthWriteEnabled: d.depthEnabled && d.writeDepth,
            depthCompare: d.depthEnabled ? (toGpuCompareFunction(d.depthFunc) ?? "always") : "always",
            stencilFront: face(d.stencilFront),
            stencilBack: face(d.stencilBack),
            stencilReadMask: d.stencilReadMask,
            stencilWriteMask: d.stencilWriteMask,
        };
    }
}
