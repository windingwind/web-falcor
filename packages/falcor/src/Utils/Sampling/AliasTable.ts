/**
 * Mirrors Falcor/Utils/Sampling/AliasTable: Vose's O(N) alias table over
 * float weights, uploaded as `items` (threshold, indexA, indexB, pad) and
 * `weights` buffers for Utils/Sampling/AliasTable.slang.
 */

import type { Device } from "../../Core/API/Device.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { RuntimeError } from "../../Core/Error.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";

const kInvalid = 0xffffffff;

export class AliasTable {
    private readonly count: number;
    private readonly weightSum: number;
    private readonly items: Buffer;
    private readonly weights: Buffer;

    /** `rng` is accepted for API parity; native constructs a distribution from it but never draws. */
    constructor(device: Device, weightsIn: ArrayLike<number>, _rng?: unknown) {
        const weights = Float32Array.from(weightsIn);
        this.count = weights.length;
        if (this.count >= kInvalid) throw new RuntimeError("Too many entries for alias table.");
        const makeBuffer = (data: ArrayBufferView, structSize: number, name: string) => {
            const b = new Buffer(device, { size: Math.max(1, this.count) * structSize, structSize, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name });
            b.setBlob(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
            return b;
        };
        this.weights = makeBuffer(weights.slice(), 4, "AliasTable::weights");

        const lowIdx = new Uint32Array(this.count).fill(kInvalid);
        const highIdx = new Uint32Array(this.count).fill(kInvalid);
        let sum = 0; // double, as native
        for (const f of weights) sum += f;
        this.weightSum = sum;
        const avgWeight = Math.fround(sum / this.count);
        let lowCount = 0;
        let highCount = 0;
        for (let i = 0; i < this.count; i++) {
            if (weights[i]! < avgWeight) lowIdx[lowCount++] = i;
            else highIdx[highCount++] = i;
        }

        const items = new ArrayBuffer(Math.max(1, this.count) * 16);
        const f32 = new Float32Array(items);
        const u32 = new Uint32Array(items);
        const set = (i: number, threshold: number, a: number, b: number) => {
            f32[i * 4] = threshold;
            u32[i * 4 + 1] = a;
            u32[i * 4 + 2] = b;
        };
        for (let i = 0; i < this.count; i++) {
            const lo = lowIdx[i]!;
            const hi = highIdx[i]!;
            if (lo !== kInvalid && hi !== kInvalid) {
                // Merge an under- and an overweighted entry; the residual re-enters a list.
                set(i, Math.fround(weights[lo]! / avgWeight), hi, lo);
                const updatedWeight = Math.fround(Math.fround(weights[lo]! + weights[hi]!) - avgWeight);
                weights[hi] = updatedWeight;
                if (updatedWeight < avgWeight) lowIdx[lowCount++] = hi;
                else highIdx[highCount++] = hi;
            } else if (hi !== kInvalid) {
                set(i, 1, hi, hi);
            } else if (lo !== kInvalid) {
                set(i, 1, lo, lo);
            } else {
                throw new RuntimeError("AliasTable: construction left an entry without an index");
            }
        }
        this.items = makeBuffer(new Uint8Array(items), 16, "AliasTable::items");
    }

    getCount(): number {
        return this.count;
    }

    getWeightSum(): number {
        return this.weightSum;
    }

    /** Mirrors AliasTable::bindShaderData. */
    bindShaderData(var_: ShaderVar): void {
        var_["items"] = this.items;
        var_["weights"] = this.weights;
        var_["count"] = this.count;
        var_["weightSum"] = Math.fround(this.weightSum);
    }
}
