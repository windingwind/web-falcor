/**
 * Parallel reduction mirroring Falcor/Utils/Algorithm/ParallelReduction.h.
 * Two-pass tiled reduction over a texture; result read back asynchronously
 * (async divergence, DESIGN.md §9).
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import type { Texture } from "../../Core/API/Texture.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { ResourceBindFlags } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import { ResourceFormat } from "../../Core/API/Formats.js";

const kShaderFile = "Utils/Algorithm/ParallelReduction.cs.slang";

/** Mirrors ParallelReduction::Type. */
export enum ParallelReductionType {
    Sum = 1, // REDUCTION_TYPE_SUM
    MinMax = 2, // REDUCTION_TYPE_MINMAX
}

enum FormatType {
    Float = 1,
    Sint = 2,
    Uint = 3,
}

function getFormatType(format: ResourceFormat): FormatType {
    const name = ResourceFormat[format]!;
    if (name.includes("Int") && !name.includes("Uint")) return FormatType.Sint;
    if (name.includes("Uint")) return FormatType.Uint;
    return FormatType.Float;
}

export class ParallelReduction {
    private passes = new Map<string, { initial: ComputePass; final: ComputePass }>();
    private buffers: [Buffer | null, Buffer | null] = [null, null];
    private allocatedElems = 0;

    constructor(public readonly device: Device) {}

    private getPasses(reductionType: ParallelReductionType, formatType: FormatType): { initial: ComputePass; final: ComputePass } {
        const key = `${reductionType}:${formatType}`;
        let p = this.passes.get(key);
        if (!p) {
            const defines = { REDUCTION_TYPE: reductionType, FORMAT_CHANNELS: 4, FORMAT_TYPE: formatType };
            p = {
                initial: ComputePass.create(this.device, { path: kShaderFile, csEntry: "initialPass", defines }),
                final: ComputePass.create(this.device, { path: kShaderFile, csEntry: "finalPass", defines }),
            };
            this.passes.set(key, p);
        }
        return p;
    }

    private allocate(elementCount: number): void {
        if (this.allocatedElems >= elementCount) return;
        // Each element is a float4/int4/uint4 pair slot (MinMax needs 2x).
        const size = elementCount * 2 * 16;
        this.buffers = [
            new Buffer(this.device, { size, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, name: "ParallelReduction[0]" }),
            new Buffer(this.device, { size, structSize: 16, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, name: "ParallelReduction[1]" }),
        ];
        this.allocatedElems = elementCount;
    }

    /**
     * Mirrors ParallelReduction::execute<float4>: reduces a texture, returns
     * 4 values (Sum) or 8 values (MinMax: min.xyzw, max.xyzw).
     */
    async execute(ctx: RenderContext, input: Texture, type: ParallelReductionType): Promise<Float32Array | Int32Array | Uint32Array> {
        const formatType = getFormatType(input.format);
        const { initial, final } = this.getPasses(type, formatType);

        const numTilesX = Math.ceil(input.width / 32);
        const numTilesY = Math.ceil(input.height / 32);
        let elems = numTilesX * numTilesY;
        this.allocate(Math.ceil(elems / 1024) * 1024 || 1024);

        const valueMult = type === ParallelReductionType.MinMax ? 2 : 1;

        {
            // WGSL emission declares only statically-used bindings; initialPass uses gInput/gResult.
            const root = initial.getRootVar();
            root["PerFrameCB"]["gResolution"] = [input.width, input.height];
            root["PerFrameCB"]["gNumTiles"] = [numTilesX, numTilesY];
            root["gInput"] = input;
            root["gResult"] = this.buffers[0]!;
            this.dispatchGroups(ctx, initial, numTilesX, numTilesY);
        }

        let inputIdx = 0;
        while (elems > 1) {
            const groups = Math.ceil(elems / 1024);
            const root = final.getRootVar();
            root["PerFrameCB"]["gElems"] = elems;
            root["gInputBuffer"] = this.buffers[inputIdx]!;
            root["gResult"] = this.buffers[1 - inputIdx]!;
            this.dispatchGroups(ctx, final, groups, 1);
            inputIdx = 1 - inputIdx;
            elems = groups;
        }

        const bytes = await ctx.readBuffer(this.buffers[inputIdx]!, 0, 16 * valueMult);
        if (formatType === FormatType.Sint) return new Int32Array(bytes.buffer, 0, 4 * valueMult);
        if (formatType === FormatType.Uint) return new Uint32Array(bytes.buffer, 0, 4 * valueMult);
        return new Float32Array(bytes.buffer, 0, 4 * valueMult);
    }

    /** Dispatches raw group counts (ComputePass.execute takes thread counts). */
    private dispatchGroups(ctx: RenderContext, pass: ComputePass, groupsX: number, groupsY: number): void {
        const [gx, gy] = pass.getThreadGroupSize();
        pass.execute(ctx, groupsX * gx, groupsY * gy, 1);
    }
}
