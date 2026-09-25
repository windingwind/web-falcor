/**
 * Port of Rendering/Materials/RGLAcquisition: a virtual measurement of a scene material's BRDF in
 * the Dupuy & Jakob parametrization ("An Adaptive Parameterization for Efficient Material
 * Acquisition and Rendering"), with native's RGLAcquisition.cs.slang kernels: retroreflection ->
 * Fredholm kernel -> NDF by power iteration -> projected areas (sigma) -> theta table -> VNDF made
 * samplable -> luminance/RGB tables. toRGLFile() gives the fields an RGL `.bsdf` file holds.
 * §9: GPU readbacks are asynchronous, so acquireIsotropic and toRGLFile return promises.
 */

import type { Device } from "../../Core/API/Device.js";
import type { Buffer } from "../../Core/API/Buffer.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import { ResourceBindFlags } from "../../Core/API/Types.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import { RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";
import type { Scene } from "../../Scene/Scene.js";
import { buildSamplableDistribution4D, RGLFieldType, type RGLField } from "../../Scene/Material/RGLFile.js";

const kShaderFile = "Rendering/Materials/RGLAcquisition.cs.slang";
const kPhiSize = 1; // isotropic only
const kThetaSize = 8;
const kNDFSize: [number, number] = [128, 2];
const kSigmaIntegrationGrid: [number, number] = [128, 128];
const kVNDFSize: [number, number, number, number] = [kPhiSize, kThetaSize, 128, 128];
const kLumiSize: [number, number, number, number] = [kPhiSize, kThetaSize, 32, 32];
const kEigenVectorPowerIterations = 4;
/** Native's lower bound on the wave size, which sizes the partial-sigma buffer. */
const kMinWaveSize = 16;
const kNDFN = kNDFSize[0] * kNDFSize[1];
const kSigmaIntegrationN = kSigmaIntegrationGrid[0] * kSigmaIntegrationGrid[1];
const kLumiN = kLumiSize.reduce((a, b) => a * b, 1);
const kVNDFN = kVNDFSize.reduce((a, b) => a * b, 1);

const kEntryPoints = ["measureRetroreflection", "buildPowerIterationKernel", "powerIteration", "integrateSigma", "sumSigma", "computeTheta", "computeVNDF", "acquireBRDF"] as const;
type EntryPoint = (typeof kEntryPoints)[number];

export class RGLAcquisition {
    private readonly passes = new Map<EntryPoint, ComputePass>();
    private readonly buffers: Record<string, Buffer> = {};
    private ndf: Buffer;
    private ndfTmp: Buffer;

    constructor(
        private readonly device: Device,
        private readonly scene: Scene,
    ) {
        const defines = scene.getSceneDefines();
        for (const e of kEntryPoints) this.passes.set(e, ComputePass.create(device, { path: kShaderFile, csEntry: e, defines }));
        const make = (elemSize: number, count: number) => device.createStructuredBuffer(elemSize, count, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
        // float3 elements take 16 bytes in WGSL storage layout.
        this.buffers["ndfDirections"] = make(16, kNDFN);
        this.buffers["retroReflection"] = make(4, kNDFN);
        this.buffers["ndfKernel"] = make(4, kNDFN * kNDFN);
        this.ndfTmp = make(4, kNDFN);
        this.ndf = make(4, kNDFN);
        this.buffers["sigma"] = make(4, (kSigmaIntegrationN * kNDFN) / kMinWaveSize);
        this.buffers["thetas"] = make(4, kThetaSize);
        this.buffers["phis"] = make(4, kPhiSize);
        this.buffers["vndf"] = make(4, kVNDFN);
        this.buffers["lumi"] = make(4, kLumiN);
        this.buffers["rgb"] = make(4, kLumiN * 3);
        this.buffers["vndfConditionalBuf"] = make(4, kVNDFN);
        this.buffers["vndfMarginalBuf"] = make(4, kVNDFN / kVNDFSize[2]);
    }

    private bind(pass: ComputePass, materialID: number): void {
        const root = pass.getRootVar();
        const v = root["gAcquisition"] as ShaderVar;
        v["materialID"] = materialID;
        v["ndfSize"] = kNDFSize;
        v["phiSize"] = kPhiSize;
        v["vndfSize"] = kVNDFSize;
        v["lumiSize"] = kLumiSize;
        v["thetaSize"] = kThetaSize;
        v["sigmaIntegrationGrid"] = kSigmaIntegrationGrid;
        for (const [name, buf] of Object.entries(this.buffers)) if (name !== "vndfConditionalBuf" && name !== "vndfMarginalBuf") v[name] = buf;
        v["ndf"] = this.ndf;
        v["ndfTmp"] = this.ndfTmp;
        // The VNDF is read back through ByteAddressBuffers named after the vndf table.
        v["vndfBuf"] = this.buffers["vndf"]!;
        v["vndfMarginalBuf"] = this.buffers["vndfMarginalBuf"]!;
        v["vndfConditionalBuf"] = this.buffers["vndfConditionalBuf"]!;
        this.scene.bindShaderData(root);
    }

    private run(ctx: RenderContext, e: EntryPoint, materialID: number, x: number, y: number, z: number): void {
        const pass = this.passes.get(e)!;
        this.bind(pass, materialID);
        pass.execute(ctx, x, y, z);
    }

    /** Mirrors RGLAcquisition::acquireIsotropic. */
    async acquireIsotropic(ctx: RenderContext, materialID: number): Promise<void> {
        if (!(materialID >= 0 && materialID < this.scene.getMaterialCount())) throw new RuntimeError("'materialID' is out of range");
        const t0 = performance.now();
        const floats = async (b: Buffer, count?: number) => new Float32Array((await b.getBlob()).buffer).slice(0, count);

        // Phase 1: retroreflection, the Fredholm kernel, and its largest eigenvector (the NDF).
        this.run(ctx, "measureRetroreflection", materialID, kNDFSize[0], kNDFSize[1], 1);
        this.run(ctx, "buildPowerIterationKernel", materialID, kNDFN, kNDFN, 1);
        this.ndf.setBlob(new Float32Array(kNDFN).fill(1));
        for (let i = 0; i < kEigenVectorPowerIterations; i++) {
            this.run(ctx, "powerIteration", materialID, kNDFSize[0], kNDFSize[1], 1);
            [this.ndf, this.ndfTmp] = [this.ndfTmp, this.ndf];
        }

        // Phase 2: projected microfacet areas; normalize the NDF and sigma rows by sigma at theta_i = 0.
        this.run(ctx, "integrateSigma", materialID, kSigmaIntegrationGrid[0], kSigmaIntegrationGrid[1], kNDFN);
        this.run(ctx, "sumSigma", materialID, kNDFSize[0], kNDFSize[1], 1);
        const ndf = await floats(this.ndf, kNDFN);
        const sigma = await floats(this.buffers["sigma"]!, kNDFN);
        for (let y = 0; y < kNDFSize[1]; y++) {
            const norm = Math.fround(1 / sigma[y * kNDFSize[0]]!);
            for (let x = 0; x < kNDFSize[0]; x++) {
                sigma[x + y * kNDFSize[0]] = Math.fround(sigma[x + y * kNDFSize[0]]! * norm);
                ndf[x + y * kNDFSize[0]] = Math.fround(ndf[x + y * kNDFSize[0]]! * norm);
            }
        }
        this.buffers["sigma"]!.setBlob(sigma);
        this.ndf.setBlob(ndf);

        // Phase 3: incident angles; phase 4: the VNDF, made samplable (SamplableDistribution4D).
        this.run(ctx, "computeTheta", materialID, kThetaSize, 1, 1);
        this.run(ctx, "computeVNDF", materialID, kVNDFSize[0], kVNDFSize[1], kVNDFSize[2] * kVNDFSize[3]);
        const dist = buildSamplableDistribution4D(await floats(this.buffers["vndf"]!, kVNDFN), kVNDFSize);
        this.buffers["vndf"]!.setBlob(dist.pdf);
        this.buffers["vndfConditionalBuf"]!.setBlob(dist.conditional);
        this.buffers["vndfMarginalBuf"]!.setBlob(dist.marginal);

        // Phase 5: the measurement.
        this.run(ctx, "acquireBRDF", materialID, kLumiSize[0], kLumiSize[1], kLumiSize[2] * kLumiSize[3]);
        await this.buffers["lumi"]!.getBlob(0, 4);
        Logger.info(`Finished BSDF acquisition in ${((performance.now() - t0) / 1000).toFixed(3)} seconds.`);
    }

    /** Mirrors RGLAcquisition::toRGLFile: the fields of a measured `.bsdf` file. */
    async toRGLFile(): Promise<RGLField[]> {
        const read = async (b: Buffer, n: number) => new Float32Array((await b.getBlob()).buffer).slice(0, n);
        const description = "Virtually measured BRDF";
        return [
            { name: "description", type: RGLFieldType.UInt8, shape: [description.length], data: new TextEncoder().encode(description) },
            { name: "phi_i", type: RGLFieldType.Float32, shape: [kPhiSize], data: await read(this.buffers["phis"]!, kPhiSize) },
            { name: "theta_i", type: RGLFieldType.Float32, shape: [kThetaSize], data: await read(this.buffers["thetas"]!, kThetaSize) },
            { name: "sigma", type: RGLFieldType.Float32, shape: [kNDFSize[1], kNDFSize[0]], data: await read(this.buffers["sigma"]!, kNDFN) },
            { name: "ndf", type: RGLFieldType.Float32, shape: [kNDFSize[1], kNDFSize[0]], data: await read(this.ndf, kNDFN) },
            { name: "vndf", type: RGLFieldType.Float32, shape: [...kVNDFSize], data: await read(this.buffers["vndf"]!, kVNDFN) },
            { name: "luminance", type: RGLFieldType.Float32, shape: [...kLumiSize], data: await read(this.buffers["lumi"]!, kLumiN) },
            { name: "rgb", type: RGLFieldType.Float32, shape: [kLumiSize[0], kLumiSize[1], 3, kLumiSize[2], kLumiSize[3]], data: await read(this.buffers["rgb"]!, kLumiN * 3) },
        ];
    }
}
