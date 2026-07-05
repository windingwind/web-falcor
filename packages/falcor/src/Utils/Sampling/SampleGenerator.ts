/**
 * GPU sample generator selector mirroring Falcor/Utils/Sampling/SampleGenerator.{h,cpp}.
 * The generator itself lives in shader code (Utils/Sampling/SampleGenerator.slang);
 * the host object only supplies the SAMPLE_GENERATOR_TYPE define. Neither
 * generator type carries GPU state, so bindShaderData is a no-op (as upstream).
 */

import type { Device } from "../../Core/API/Device.js";
import { DefineList } from "../../Core/Program/DefineList.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { ArgumentError } from "../../Core/Error.js";

/** Mirrors Utils/Sampling/SampleGeneratorType.slangh. */
export const SAMPLE_GENERATOR_TINY_UNIFORM = 0;
export const SAMPLE_GENERATOR_UNIFORM = 1;
export const SAMPLE_GENERATOR_DEFAULT = SAMPLE_GENERATOR_UNIFORM;

export class SampleGenerator {
    private constructor(private readonly type: number) {}

    static create(_device: Device, type: number): SampleGenerator {
        if (type !== SAMPLE_GENERATOR_TINY_UNIFORM && type !== SAMPLE_GENERATOR_UNIFORM) {
            throw new ArgumentError("Can't create SampleGenerator. Unknown type");
        }
        return new SampleGenerator(type);
    }

    getDefines(): DefineList {
        return new DefineList().add("SAMPLE_GENERATOR_TYPE", String(this.type));
    }

    bindShaderData(_var: ShaderVar): void {}
}
