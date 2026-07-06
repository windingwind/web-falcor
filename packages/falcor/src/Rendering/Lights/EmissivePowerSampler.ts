/**
 * Power-weighted emissive light sampler mirroring
 * Falcor/Rendering/Lights/EmissivePowerSampler.{h,cpp}. The alias table build
 * replicates native bit-for-bit: normalization, ascending-weight permutation
 * sort, head/tail merging, an mt19937-driven shuffle (default seed, full-range
 * draws — matching libstdc++ uniform_int_distribution over the full uint32
 * range) and the f16-threshold + 2x24-bit index packing.
 */

import type { Device } from "../../Core/API/Device.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { f32tof16 } from "../../Scene/Material/MaterialData.js";

/** std::mt19937 (32-bit Mersenne Twister), default seed 5489. */
class Mt19937 {
    private mt = new Uint32Array(624);
    private index = 625;

    constructor(seed = 5489) {
        this.mt[0] = seed >>> 0;
        for (let i = 1; i < 624; i++) {
            const prev = this.mt[i - 1]! ^ (this.mt[i - 1]! >>> 30);
            // 1812433253 * prev + i, in 32-bit arithmetic (split to avoid f64 precision loss).
            const lo = (prev & 0xffff) * 1812433253;
            const hi = (((prev >>> 16) * 1812433253) & 0xffff) << 16;
            this.mt[i] = (((lo + hi) >>> 0) + i) >>> 0;
        }
        this.index = 624;
    }

    next(): number {
        if (this.index >= 624) {
            for (let i = 0; i < 624; i++) {
                const y = ((this.mt[i]! & 0x80000000) | (this.mt[(i + 1) % 624]! & 0x7fffffff)) >>> 0;
                let next = (this.mt[(i + 397) % 624]! ^ (y >>> 1)) >>> 0;
                if (y & 1) next = (next ^ 0x9908b0df) >>> 0;
                this.mt[i] = next;
            }
            this.index = 0;
        }
        let y = this.mt[this.index++]!;
        y = (y ^ (y >>> 11)) >>> 0;
        y = (y ^ ((y << 7) & 0x9d2c5680)) >>> 0;
        y = (y ^ ((y << 15) & 0xefc60000)) >>> 0;
        return (y ^ (y >>> 18)) >>> 0;
    }
}

export class EmissivePowerSampler {
    private readonly table: Buffer;
    private readonly invWeightsSum: number;
    private readonly rng = new Mt19937();

    /** Builds the alias table from per-triangle flux (LightCollection order). */
    constructor(device: Device, fluxes: Float32Array) {
        const N = fluxes.length;
        const weights = Float32Array.from(fluxes);

        let sum = 0; // double accumulation, as native
        for (const f of weights) sum += f;
        const scale = Math.fround(N / Math.fround(sum)); // f *= N / float(sum)
        for (let i = 0; i < N; i++) weights[i] = Math.fround(weights[i]! * scale);

        const permutation = Array.from({ length: N }, (_v, i) => i);
        permutation.sort((a, b) => weights[a]! - weights[b]!);

        const thresholds = new Float32Array(N);
        const redirect = new Uint32Array(N);

        let head = 0;
        let tail = N - 1;
        while (head !== tail) {
            const i = permutation[head]!;
            const j = permutation[tail]!;
            thresholds[i] = weights[i]!;
            redirect[i] = j;
            weights[j] = Math.fround(weights[j]! - Math.fround(1 - weights[i]!));
            if (head === tail - 1) {
                thresholds[j] = 1;
                redirect[j] = j;
                break;
            } else if (weights[j]! < 1) {
                const t = permutation[head]!;
                permutation[head] = permutation[tail]!;
                permutation[tail] = t;
                tail--;
            } else {
                head++;
            }
        }
        if (N === 1) {
            thresholds[0] = 1;
            redirect[0] = 0;
        }

        const perm2 = Array.from({ length: N }, (_v, i) => i);
        for (let i = 0; i < N; i++) {
            const dst = i + (this.rng.next() % (N - i));
            const swap = <T>(arr: { [k: number]: T }) => {
                const t = arr[i]!;
                arr[i] = arr[dst]!;
                arr[dst] = t;
            };
            swap(thresholds);
            swap(redirect);
            swap(perm2);
        }

        const packed = new Uint32Array(N * 2);
        for (let i = 0; i < N; i++) {
            const prob = (f32tof16(thresholds[i]!) << 16) >>> 0;
            const lowX = redirect[i]! & 0xffffff;
            const lowY = perm2[i]! & 0xffffff;
            packed[i * 2] = (prob | ((lowX >>> 8) & 0xffff)) >>> 0;
            packed[i * 2 + 1] = (((lowX & 0xff) << 24) | lowY) >>> 0;
        }

        this.invWeightsSum = 1 / sum;
        this.table = new Buffer(device, {
            size: Math.max(N, 1) * 8,
            structSize: 8,
            bindFlags: ResourceBindFlags.ShaderResource,
            memoryType: MemoryType.DeviceLocal,
            name: "EmissivePowerSampler::aliasTable",
        });
        this.table.setBlob(new Uint8Array(packed.buffer));
    }

    bindShaderData(var_: ShaderVar): void {
        const p = var_["_emissivePower"] as ShaderVar;
        p["invWeightsSum"] = this.invWeightsSum;
        p["triangleAliasTable"] = this.table;
    }
}
