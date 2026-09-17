/**
 * IES light profile mirroring Scene/Lights/LightProfile.{h,cpp}.
 *
 * Reads an IESNA LM-63 photometric file, then bakes it into the 256x256 R16F
 * table the shader samples, with the flux factor reduced on the GPU — the same
 * two steps native performs (`createFromIesProfile` + `bake`).
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { Texture } from "../../Core/API/Texture.js";
import { Sampler, TextureFilteringMode, TextureAddressingMode } from "../../Core/API/Sampler.js";
import { MemoryType, ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import { ParallelReduction, ParallelReductionType } from "../../Utils/Algorithm/ParallelReduction.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";

const kBakeShaderFile = "Scene/Lights/BakeIesProfile.cs.slang";
/** Mirrors LightProfile.cpp's kBakeResolution. */
export const kBakeResolution = 256;

/** Profile identifiers native accepts on the first line of the file. */
const kSupportedProfiles = [
    "IESNA:LM-63-1986",
    "IESNA:LM-63-1991",
    "IESNA91",
    "IESNA:LM-63-1995",
    "IESNA:LM-63-2002",
    "ERCO Leuchten GmbH  BY: ERCO/LUM650/8701",
    "ERCO Leuchten GmbH",
];

/** Number of leading values before the angle tables (see LightProfile.cpp). */
const kHeaderSize = 13;

/**
 * Mirrors parseIesFile: validates the profile identifier and the TILT line,
 * then reads every remaining whitespace-separated number.
 *
 * @returns The numeric block, with `data[0]` replaced by the normalization
 *          factor (native reuses that slot, which rendering does not need).
 */
export function parseIesProfile(text: string, normalize: boolean): Float32Array {
    const lines = text.split(/\r\n|\r|\n/);
    if (!lines.length || !kSupportedProfiles.some((p) => lines[0]!.includes(p))) {
        throw new RuntimeError("LightProfile: unsupported IES profile");
    }

    // Header keywords run until the TILT line; only TILT=NONE is supported.
    let dataStart = -1;
    for (let i = 1; i < lines.length; i++) {
        const line = lines[i]!;
        if (line.startsWith("TILT=NONE")) {
            dataStart = i + 1;
            break;
        }
        if (line.startsWith("TILT=")) throw new RuntimeError("LightProfile: TILT profiles are unsupported");
    }
    if (dataStart < 0) throw new RuntimeError("LightProfile: no TILT line found");

    const numbers: number[] = [];
    for (const token of lines.slice(dataStart).join(" ").split(/[\s,]+/)) {
        if (token === "") continue;
        const value = Number.parseFloat(token);
        if (Number.isFinite(value)) numbers.push(value);
    }
    if (numbers.length < 16) throw new RuntimeError("LightProfile: IES data block too short");

    const verticalCount = Math.trunc(numbers[3]!);
    const horizontalCount = Math.trunc(numbers[4]!);
    const expected = kHeaderSize + horizontalCount + verticalCount + horizontalCount * verticalCount;
    if (numbers.length !== expected) {
        throw new RuntimeError(`LightProfile: IES data size mismatch (got ${numbers.length}, expected ${expected})`);
    }

    let maxCandelas = 0;
    for (let i = kHeaderSize + horizontalCount + verticalCount; i < expected; i++) maxCandelas = Math.max(maxCandelas, numbers[i]!);

    const data = Float32Array.from(numbers);
    // Native stashes the normalization factor in data[0] (the lamp count is unused).
    data[0] = normalize ? 1 / maxCandelas : 1;
    return data;
}

export class LightProfile {
    private texture: Texture | null = null;
    private sampler: Sampler | null = null;
    private fluxFactorValue = 0;

    private constructor(
        private readonly device: Device,
        readonly name: string,
        /** The parsed IES numbers, as the bake kernel indexes them. */
        readonly rawData: Float32Array,
    ) {}

    /** Mirrors LightProfile::createFromIesProfile (web: the file arrives over fetch). */
    static async createFromIesProfile(device: Device, url: string, normalize = true): Promise<LightProfile> {
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`LightProfile: can't open file '${url}'`);
        const name = url.split("/").pop() ?? url;
        return new LightProfile(device, name, parseIesProfile(await res.text(), normalize));
    }

    /** Mirrors LightProfile::getFluxFactor (valid after bake). */
    get fluxFactor(): number {
        return this.fluxFactorValue;
    }

    getTexture(): Texture | null {
        return this.texture;
    }

    /**
     * Mirrors LightProfile::bake: evaluates the profile over a 256x256
     * (vertical, horizontal) grid and reduces the flux texture to a single
     * factor. Async here because the readback is (docs §9).
     */
    async bake(ctx: RenderContext): Promise<void> {
        const dataBuffer = new Buffer(this.device, {
            size: this.rawData.byteLength,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource,
            memoryType: MemoryType.DeviceLocal,
            name: "LightProfile::iesData",
        });
        dataBuffer.setBlob(new Uint8Array(this.rawData.buffer, this.rawData.byteOffset, this.rawData.byteLength));

        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        this.texture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: kBakeResolution,
            height: kBakeResolution,
            // §9: R16Float is not a WGSL storage format; the profile is baked at
            // full precision instead (the sampled values are identical to 1e-3).
            format: ResourceFormat.R32Float,
            bindFlags: storage,
            mipLevels: 1,
            name: "LightProfile::texture",
        });
        const fluxTexture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: kBakeResolution,
            height: kBakeResolution,
            format: ResourceFormat.R32Float,
            bindFlags: storage,
            mipLevels: 1,
            name: "LightProfile::fluxTexture",
        });

        const pass = ComputePass.create(this.device, { path: kBakeShaderFile });
        const root = pass.getRootVar();
        root["gIesData"] = dataBuffer;
        root["gTexture"] = this.texture;
        root["gFluxTexture"] = fluxTexture;
        (root["CB"] as ShaderVar)["gBakeResolution"] = kBakeResolution;
        pass.execute(ctx, kBakeResolution, kBakeResolution);

        const reduction = new ParallelReduction(this.device);
        const sum = await reduction.execute(ctx, fluxTexture, ParallelReductionType.Sum);
        this.fluxFactorValue = sum[0]!;

        this.sampler = new Sampler(this.device, {
            minFilter: TextureFilteringMode.Linear,
            magFilter: TextureFilteringMode.Linear,
            mipFilter: TextureFilteringMode.Linear,
            addressModeU: TextureAddressingMode.Clamp,
            addressModeV: TextureAddressingMode.Clamp,
            addressModeW: TextureAddressingMode.Clamp,
        });
        Logger.info(`Baked light profile '${this.name}' (flux factor ${this.fluxFactorValue}).`);
    }

    /** Mirrors LightProfile::bindShaderData. */
    bindShaderData(var_: ShaderVar): void {
        if (!this.texture || !this.sampler) throw new RuntimeError("LightProfile: bake() must run before binding");
        var_["fluxFactor"] = this.fluxFactorValue;
        var_["texture"] = this.texture;
        var_["sampler"] = this.sampler;
    }
}
