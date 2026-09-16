/**
 * BSDF integration utility mirroring Rendering/Materials/BSDFIntegrator.h:
 * integrates an isotropic BSDF over the hemisphere on the GPU (512x512 grid,
 * 8x8 stratified samples per cell), one grid per incident direction.
 * Web divergence (docs §9): readback is async.
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { ArgumentError, RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";
import { float3 } from "../../Utils/Math/Vector.js";
import type { Scene } from "../../Scene/Scene.js";

const kShaderFile = "Rendering/Materials/BSDFIntegrator.cs.slang";
const kParameterBlock = "gIntegrator";
/** Integration grid size; the kernels are specialized for it (32x32 groups). */
const kGridSize: [number, number] = [512, 512];
/** float3 in a WGSL storage array has a 16-byte stride. */
const kFloat3Stride = 16;

export class BSDFIntegrator {
    private readonly integrationPass: ComputePass;
    private readonly finalPass: ComputePass;
    private cosThetaBuffer: Buffer | null = null;
    private resultBuffer: Buffer | null = null;
    private finalResultBuffer: Buffer | null = null;
    /** Number of intermediate results per integration grid (one per thread group). */
    readonly resultCount: number;

    constructor(
        private readonly device: Device,
        private readonly scene: Scene,
    ) {
        const defines = scene.getSceneDefines();
        this.integrationPass = ComputePass.create(device, { path: kShaderFile, csEntry: "mainIntegration", defines });
        this.finalPass = ComputePass.create(device, { path: kShaderFile, csEntry: "mainFinal", defines });

        const [gx, gy, gz] = this.integrationPass.getThreadGroupSize();
        if (gx !== 32 || gy !== 32 || gz !== 1) throw new RuntimeError("BSDFIntegrator: unexpected integration group size");
        this.resultCount = (kGridSize[0] * kGridSize[1]) / (gx * gy * gz);
        const [fx] = this.finalPass.getThreadGroupSize();
        if (fx !== this.resultCount) throw new RuntimeError("BSDFIntegrator: final group size must equal resultCount");
    }

    /** Mirrors integrateIsotropic(cosTheta) — one incident direction. */
    async integrateIsotropicSingle(ctx: RenderContext, materialID: number, cosTheta: number): Promise<float3> {
        return (await this.integrateIsotropic(ctx, materialID, [cosTheta]))[0]!;
    }

    /**
     * Mirrors integrateIsotropic(cosThetas): integrates the BSDF over outgoing
     * directions in the upper hemisphere for each incident cos(theta).
     */
    async integrateIsotropic(ctx: RenderContext, materialID: number, cosThetas: readonly number[]): Promise<float3[]> {
        if (!(materialID >= 0 && materialID < this.scene.getMaterialCount())) throw new ArgumentError("'materialID' is out of range");
        if (cosThetas.length === 0) throw new ArgumentError("'cosThetas' array is empty");
        const t0 = performance.now();
        const gridCount = cosThetas.length;

        // Upload cos theta angles.
        if (!this.cosThetaBuffer || this.cosThetaBuffer.elementCount < gridCount) {
            this.cosThetaBuffer = new Buffer(this.device, {
                size: gridCount * 4,
                structSize: 4,
                bindFlags: ResourceBindFlags.ShaderResource,
                memoryType: MemoryType.DeviceLocal,
                name: "BSDFIntegrator::cosThetas",
            });
        }
        this.cosThetaBuffer.setBlob(new Float32Array(cosThetas));

        // Allocate buffers for intermediate and final results.
        const elemCount = gridCount * this.resultCount;
        if (!this.resultBuffer || this.resultBuffer.elementCount < elemCount) {
            this.resultBuffer = this.createFloat3Buffer(elemCount, "BSDFIntegrator::results");
        }
        if (!this.finalResultBuffer || this.finalResultBuffer.elementCount < gridCount) {
            this.finalResultBuffer = this.createFloat3Buffer(gridCount, "BSDFIntegrator::finalResults");
        }

        this.integrationPassExec(ctx, materialID, gridCount);
        this.finalPassExec(ctx, gridCount);

        // Read back final results (async divergence, docs §9).
        const bytes = await this.finalResultBuffer.getBlob(0, gridCount * kFloat3Stride);
        const data = new Float32Array(bytes.buffer, bytes.byteOffset, gridCount * 4);
        const output: float3[] = [];
        for (let i = 0; i < gridCount; i++) output.push(new float3(data[i * 4]!, data[i * 4 + 1]!, data[i * 4 + 2]!));
        Logger.info(`Finished BSDF integration for ${gridCount} incident directions in ${((performance.now() - t0) / 1000).toFixed(3)} seconds.`);
        return output;
    }

    private createFloat3Buffer(elementCount: number, name: string): Buffer {
        return new Buffer(this.device, {
            size: elementCount * kFloat3Stride,
            structSize: kFloat3Stride,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name,
        });
    }

    private integrationPassExec(ctx: RenderContext, materialID: number, gridCount: number): void {
        const root = this.integrationPass.getRootVar();
        const v = root[kParameterBlock] as ShaderVar;
        v["gridSize"] = kGridSize;
        v["gridCount"] = gridCount;
        v["resultCount"] = this.resultCount;
        v["materialID"] = materialID;
        v["cosThetas"] = this.cosThetaBuffer!;
        v["results"] = this.resultBuffer!;
        this.scene.bindShaderData(root);
        this.integrationPass.execute(ctx, kGridSize[0], kGridSize[1], gridCount);
    }

    private finalPassExec(ctx: RenderContext, gridCount: number): void {
        const root = this.finalPass.getRootVar();
        const v = root[kParameterBlock] as ShaderVar;
        v["gridCount"] = gridCount;
        v["resultCount"] = this.resultCount;
        v["results"] = this.resultBuffer!;
        v["finalResults"] = this.finalResultBuffer!;
        this.finalPass.execute(ctx, this.resultCount, gridCount, 1);
    }
}
