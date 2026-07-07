/**
 * RTXDI (ReSTIR DI) host mirroring Rendering/RTXDI/RTXDI.{h,cpp} plus the
 * rtxdi::Context parameter math from the SDK (rtxdi-sdk/src/RTXDI.cpp;
 * ReGIR stays Disabled like Falcor). Web divergences (documented):
 * localLightPdfTexture is R32Float (WGSL has no r16float storage format)
 * and the boiling filter is compiled out (see the RTXDI.slang override).
 */

import { Buffer } from "../../Core/API/Buffer.js";
import { ComputePass } from "../../Core/Pass/ComputePass.js";
import { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType, MemoryType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";
import type { Scene } from "../../Scene/Scene.js";
import type { CameraData } from "../../Scene/Camera/Camera.js";
import { LightType } from "../../Scene/SceneData.js";
import { DefineList } from "../../Core/Program/DefineList.js";

const kRTXDIShadersFile = "Rendering/RTXDI/RTXDISetup.cs.slang";
const kLightUpdaterShaderFile = "Rendering/RTXDI/LightUpdater.cs.slang";
const kEnvLightUpdaterShaderFile = "Rendering/RTXDI/EnvLightUpdater.cs.slang";

const kMaxReservoirs = 3; // Number of reservoirs allocated (last one holds per-frame candidates).
const kCandidateReservoirID = 2;
const kReservoirBlockSize = 16; // RTXDI_RESERVOIR_BLOCK_SIZE
const kNeighborOffsetCount = 8192;

// Packed struct strides (PackedTypes.slang / Reservoir.hlsli; all 4-byte fields).
const kPackedPolymorphicLightSize = 32;
const kPackedSurfaceDataSize = 32;
const kPackedReservoirSize = 24;

export type RTXDIMode = "NoResampling" | "SpatialResampling" | "TemporalResampling" | "SpatiotemporalResampling";

/** Mirrors RTXDI::Options (defaults identical). */
export interface RTXDIOptions {
    mode: RTXDIMode;
    presampledTileCount: number;
    presampledTileSize: number;
    storeCompactLightInfo: boolean;
    localLightCandidateCount: number;
    infiniteLightCandidateCount: number;
    envLightCandidateCount: number;
    brdfCandidateCount: number;
    brdfCutoff: number;
    testCandidateVisibility: boolean;
    biasCorrection: number; // 0 Off, 1 Basic, 2 Pairwise, 3 RayTraced
    depthThreshold: number;
    normalThreshold: number;
    samplingRadius: number;
    spatialSampleCount: number;
    spatialIterations: number;
    maxHistoryLength: number;
    boilingFilterStrength: number;
    rayEpsilon: number;
    useEmissiveTextures: boolean;
    enableVisibilityShortcut: boolean;
    enablePermutationSampling: boolean;
}

export const kDefaultRTXDIOptions: RTXDIOptions = {
    mode: "SpatiotemporalResampling",
    presampledTileCount: 128,
    presampledTileSize: 1024,
    storeCompactLightInfo: true,
    localLightCandidateCount: 24,
    infiniteLightCandidateCount: 8,
    envLightCandidateCount: 8,
    brdfCandidateCount: 1,
    brdfCutoff: 0,
    testCandidateVisibility: true,
    biasCorrection: 1,
    depthThreshold: 0.1,
    normalThreshold: 0.5,
    samplingRadius: 30,
    spatialSampleCount: 1,
    spatialIterations: 5,
    maxHistoryLength: 20,
    boilingFilterStrength: 0,
    rayEpsilon: 1e-3,
    useEmissiveTextures: false,
    enableVisibilityShortcut: false,
    enablePermutationSampling: false,
};

/** 32-bit Jenkins hash (rtxdi-sdk). */
function jenkinsHash(a: number): number {
    a = (a + 0x7ed55d16 + ((a << 12) >>> 0)) >>> 0;
    a = ((a ^ 0xc761c23c) ^ (a >>> 19)) >>> 0;
    a = (a + 0x165667b1 + ((a << 5) >>> 0)) >>> 0;
    a = ((a + 0xd3a2646c) ^ ((a << 9) >>> 0)) >>> 0;
    a = (a + 0xfd7046c5 + ((a << 3) >>> 0)) >>> 0;
    a = ((a ^ 0xb55a4f09) ^ (a >>> 16)) >>> 0;
    return a >>> 0;
}

/** Mirrors rtxdi::ComputePdfTextureSize. */
function computePdfTextureSize(maxItems: number): { width: number; height: number; mipLevels: number } {
    let width = Math.max(1, Math.ceil(Math.sqrt(maxItems)));
    width = Math.pow(2, Math.ceil(Math.log2(width)));
    let height = Math.max(1, Math.ceil(maxItems / width));
    height = Math.pow(2, Math.ceil(Math.log2(height)));
    const mipLevels = Math.max(1, Math.log2(Math.max(width, height)));
    return { width, height, mipLevels: Math.floor(mipLevels) };
}

/** Mirrors rtxdi::Context::FillNeighborOffsetBuffer, decoded to float (RG8Snorm -> StructuredBuffer<float2> on web). */
function fillNeighborOffsets(count: number): Float32Array {
    const out = new Float32Array(count * 2);
    const R = 250;
    const phi2 = 1.0 / 1.3247179572447;
    let num = 0;
    let u = 0.5;
    let v = 0.5;
    while (num < count * 2) {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.0) u -= 1.0;
        if (v >= 1.0) v -= 1.0;
        const rSq = (u - 0.5) * (u - 0.5) + (v - 0.5) * (v - 0.5);
        if (rSq > 0.25) continue;
        // int8 quantization exactly like native, then snorm decode (v/127 clamped).
        const x = Math.trunc((u - 0.5) * R) << 24 >> 24;
        const y = Math.trunc((v - 0.5) * R) << 24 >> 24;
        out[num++] = Math.max(x / 127, -1);
        out[num++] = Math.max(y / 127, -1);
    }
    return out;
}

export class RTXDI {
    readonly options: RTXDIOptions;

    private frameDim: [number, number] = [0, 0];
    private frameIndex = 0;
    private currentSurfaceBufferIndex = 0;
    private lastFrameReservoirID = 1;
    private prevCameraData: CameraData | null = null;
    private contextValid = false;

    // Context-derived parameters (rtxdi::Context).
    private reservoirBlockRowPitch = 0;
    private reservoirArrayPitch = 0;

    // Light bookkeeping (mirrors RTXDI::mLights).
    private localAnalyticLightCount = 0;
    private infiniteAnalyticLightCount = 0;
    private emissiveLightCount = 0;
    private envLightPresent = false;
    private prevEmissiveLightCount = -1;
    private prevLocalAnalyticLightCount = -1;
    private needsLightUpdate = true;

    // Resources.
    private lightInfoBuffer: Buffer | null = null;
    private lightTileBuffer: Buffer | null = null;
    private compactLightInfoBuffer: Buffer | null = null;
    private reservoirBuffer: Buffer | null = null;
    private surfaceDataBuffer: Buffer | null = null;
    private neighborOffsetsBuffer: Buffer | null = null;
    private analyticLightIDBuffer: Buffer | null = null;
    private localLightPdfTexture: Texture | null = null;
    private envLightLuminanceTexture: Texture | null = null;
    private envLightPdfTexture: Texture | null = null;

    // Compute passes.
    private updateLightsPass: ComputePass | null = null;
    private updateEnvLightPass: ComputePass | null = null;
    private presampleLocalLightsPass: ComputePass | null = null;
    private presampleEnvLightPass: ComputePass | null = null;
    private generateCandidatesPass: ComputePass | null = null;
    private testCandidateVisibilityPass: ComputePass | null = null;
    private spatialResamplingPass: ComputePass | null = null;
    private temporalResamplingPass: ComputePass | null = null;
    private spatiotemporalResamplingPass: ComputePass | null = null;

    constructor(
        readonly device: Device,
        readonly scene: Scene,
        options: Partial<RTXDIOptions> = {},
    ) {
        this.options = { ...kDefaultRTXDIOptions, ...options };
    }

    getDefines(): DefineList {
        return new DefineList().add("RTXDI_INSTALLED", 1);
    }

    beginFrame(ctx: RenderContext, frameDim: [number, number]): void {
        if (this.frameIndex === 0) this.prevCameraData = this.scene.camera.getData();
        if (frameDim[0] !== this.frameDim[0] || frameDim[1] !== this.frameDim[1]) {
            this.frameDim = [frameDim[0], frameDim[1]];
            this.contextValid = false;
        }
        if (!this.updateLightsPass) this.loadShaders();
        if (!this.contextValid) this.prepareResources(ctx);
    }

    endFrame(_ctx: RenderContext): void {
        this.frameIndex++;
        this.currentSurfaceBufferIndex = 1 - this.currentSurfaceBufferIndex;
        this.prevCameraData = this.scene.camera.getData();
    }

    /** Mirrors RTXDI::update: light prep, presampling and resampling. */
    update(ctx: RenderContext, motionVectors: Texture): void {
        this.updateLights(ctx);
        this.updateEnvLight(ctx);
        this.presampleLights(ctx, motionVectors);

        let outputReservoirID: number;
        switch (this.options.mode) {
            case "NoResampling":
                this.dispatchScreen(ctx, this.generateCandidatesPass!, motionVectors, { gOutputReservoirID: kCandidateReservoirID });
                outputReservoirID = kCandidateReservoirID;
                break;
            case "SpatialResampling":
                this.dispatchScreen(ctx, this.generateCandidatesPass!, motionVectors, { gOutputReservoirID: kCandidateReservoirID });
                this.testCandidateVisibility(ctx, motionVectors);
                outputReservoirID = this.spatialResampling(ctx, motionVectors, kCandidateReservoirID);
                break;
            case "TemporalResampling":
                this.dispatchScreen(ctx, this.generateCandidatesPass!, motionVectors, { gOutputReservoirID: kCandidateReservoirID });
                this.testCandidateVisibility(ctx, motionVectors);
                outputReservoirID = 1 - this.lastFrameReservoirID;
                this.dispatchScreen(ctx, this.temporalResamplingPass!, motionVectors, {
                    gTemporalReservoirID: this.lastFrameReservoirID,
                    gInputReservoirID: kCandidateReservoirID,
                    gOutputReservoirID: outputReservoirID,
                });
                break;
            case "SpatiotemporalResampling":
                this.dispatchScreen(ctx, this.generateCandidatesPass!, motionVectors, { gOutputReservoirID: kCandidateReservoirID });
                this.testCandidateVisibility(ctx, motionVectors);
                outputReservoirID = 1 - this.lastFrameReservoirID;
                this.dispatchScreen(ctx, this.spatiotemporalResamplingPass!, motionVectors, {
                    gTemporalReservoirID: this.lastFrameReservoirID,
                    gInputReservoirID: kCandidateReservoirID,
                    gOutputReservoirID: outputReservoirID,
                });
                break;
        }
        this.lastFrameReservoirID = outputReservoirID;
    }

    /** Binds gRTXDI for external passes (PrepareSurfaceData/FinalShading). */
    setShaderData(rootVar: ShaderVar, motionVectors: Texture | null = null): void {
        this.bindShaderDataInternal(rootVar, motionVectors);
    }

    // ------------------------------------------------------------------ internals

    private get localLightCount(): number {
        return this.emissiveLightCount + this.localAnalyticLightCount;
    }
    private get infiniteLightCount(): number {
        return this.infiniteAnalyticLightCount;
    }
    private get totalLightCount(): number {
        return this.localLightCount + this.infiniteLightCount + (this.envLightPresent ? 1 : 0);
    }
    private get envLightIndex(): number {
        return this.localLightCount + this.infiniteLightCount;
    }

    private loadShaders(): void {
        const create = (path: string, entry: string) => {
            const defines = this.scene.getSceneDefines().addAll(this.getDefines());
            return ComputePass.create(this.device, { path, defines, csEntry: entry });
        };
        this.updateLightsPass = create(kLightUpdaterShaderFile, "main");
        this.updateEnvLightPass = create(kEnvLightUpdaterShaderFile, "main");
        this.presampleLocalLightsPass = create(kRTXDIShadersFile, "presampleLocalLights");
        this.presampleEnvLightPass = create(kRTXDIShadersFile, "presampleEnvLight");
        this.generateCandidatesPass = create(kRTXDIShadersFile, "generateCandidates");
        this.testCandidateVisibilityPass = create(kRTXDIShadersFile, "testCandidateVisibility");
        this.spatialResamplingPass = create(kRTXDIShadersFile, "spatialResampling");
        this.temporalResamplingPass = create(kRTXDIShadersFile, "temporalResampling");
        this.spatiotemporalResamplingPass = create(kRTXDIShadersFile, "spatiotemporalResampling");
    }

    /** Mirrors RTXDI::prepareResources + rtxdi::Context construction. */
    private prepareResources(ctx: RenderContext): void {
        const tileCount = this.options.presampledTileCount;
        const tileSize = this.options.presampledTileSize;

        // rtxdi::Context: reservoir pitches (checkerboard off).
        const widthBlocks = Math.ceil(this.frameDim[0] / kReservoirBlockSize);
        const heightBlocks = Math.ceil(this.frameDim[1] / kReservoirBlockSize);
        this.reservoirBlockRowPitch = widthBlocks * kReservoirBlockSize * kReservoirBlockSize;
        this.reservoirArrayPitch = this.reservoirBlockRowPitch * heightBlocks;

        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const makeBuffer = (name: string, size: number, structSize: number) => {
            const buf = new Buffer(this.device, { size, structSize, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name });
            return buf;
        };

        // RIS light tiles (env tiles use the same count/size like Falcor).
        const risElements = Math.max(tileCount * tileSize * 2, 1);
        this.lightTileBuffer = makeBuffer("RTXDI::lightTile", risElements * 8, 8);
        this.compactLightInfoBuffer = makeBuffer("RTXDI::compactLightInfo", risElements * 2 * kPackedPolymorphicLightSize, kPackedPolymorphicLightSize);
        this.reservoirBuffer = makeBuffer("RTXDI::reservoirs", this.reservoirArrayPitch * kMaxReservoirs * kPackedReservoirSize, kPackedReservoirSize);
        ctx.clearBuffer(this.reservoirBuffer);
        this.surfaceDataBuffer = makeBuffer("RTXDI::surfaceData", 2 * this.frameDim[0] * this.frameDim[1] * kPackedSurfaceDataSize, kPackedSurfaceDataSize);

        const offsets = fillNeighborOffsets(kNeighborOffsetCount);
        this.neighborOffsetsBuffer = makeBuffer("RTXDI::neighborOffsets", offsets.byteLength, 8);
        this.neighborOffsetsBuffer.setBlob(new Uint8Array(offsets.buffer));

        this.prevEmissiveLightCount = -1;
        this.prevLocalAnalyticLightCount = -1;
        this.needsLightUpdate = true;
        this.contextValid = true;
    }

    /** Mirrors RTXDI::updateLights (web: static scenes update once). */
    private updateLights(ctx: RenderContext): void {
        // Categorize analytic lights (Point = local; Directional/Distant = infinite).
        const localIDs: number[] = [];
        const infiniteIDs: number[] = [];
        this.scene.analyticLights.forEach((light, lightID) => {
            if (light.type === LightType.Point) localIDs.push(lightID);
            else if (light.type === LightType.Directional || light.type === LightType.Distant) infiniteIDs.push(lightID);
        });
        this.localAnalyticLightCount = localIDs.length;
        this.infiniteAnalyticLightCount = infiniteIDs.length;
        this.emissiveLightCount = this.scene.useEmissiveLights ? this.scene.emissiveActiveTriangleCount : 0;
        this.envLightPresent = this.scene.useEnvLight;

        if (!this.needsLightUpdate && this.prevEmissiveLightCount === this.emissiveLightCount && this.prevLocalAnalyticLightCount === this.localAnalyticLightCount) {
            return;
        }

        const analyticIDs = new Uint32Array([...localIDs, ...infiniteIDs]);
        if (analyticIDs.length > 0) {
            this.analyticLightIDBuffer = new Buffer(this.device, {
                size: analyticIDs.byteLength,
                structSize: 4,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "RTXDI::analyticLightIDs",
            });
            this.analyticLightIDBuffer.setBlob(new Uint8Array(analyticIDs.buffer));
        }

        this.lightInfoBuffer = new Buffer(this.device, {
            size: Math.max(this.totalLightCount, 1) * kPackedPolymorphicLightSize,
            structSize: kPackedPolymorphicLightSize,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "RTXDI::lightInfo",
        });

        // Local light PDF texture (web: R32Float — WGSL has no r16float storage).
        const pdfSize = computePdfTextureSize(Math.max(this.localLightCount, 1));
        this.localLightPdfTexture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: pdfSize.width,
            height: pdfSize.height,
            mipLevels: pdfSize.mipLevels,
            format: ResourceFormat.R32Float,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget,
            name: "RTXDI::localLightPdf",
        });
        ctx.clearTexture(this.localLightPdfTexture);

        // Dispatch LightUpdater over all lights.
        const threadCountX = 8192;
        const threadCountY = Math.ceil(Math.max(this.totalLightCount, 1) / threadCountX);
        const root = this.updateLightsPass!.getRootVar();
        this.scene.bindShaderData(root);
        const v = root["gLightUpdater"] as ShaderVar;
        v["lightInfo"] = this.lightInfoBuffer;
        v["localLightPdf"] = this.localLightPdfTexture;
        if (this.analyticLightIDBuffer) v["analyticLightIDs"] = this.analyticLightIDBuffer;
        v["threadCount"] = [threadCountX, threadCountY];
        v["totalLightCount"] = this.totalLightCount;
        v["firstLocalAnalyticLight"] = this.emissiveLightCount;
        v["firstInfiniteAnalyticLight"] = this.emissiveLightCount + this.localAnalyticLightCount;
        v["envLightIndex"] = this.envLightIndex;
        v["updateEmissiveLights"] = 1;
        v["updateEmissiveLightsFlux"] = 1;
        v["updateAnalyticLights"] = 1;
        v["updateAnalyticLightsFlux"] = 1;
        this.updateLightsPass!.execute(ctx, threadCountX, threadCountY);

        this.generateMips(ctx, this.localLightPdfTexture);

        this.prevEmissiveLightCount = this.emissiveLightCount;
        this.prevLocalAnalyticLightCount = this.localAnalyticLightCount;
        this.needsLightUpdate = false;
    }

    /** Mirrors RTXDI::updateEnvLight. */
    private envLightValid = false;
    private updateEnvLight(ctx: RenderContext): void {
        if (!this.scene.useEnvLight || this.envLightValid) return;
        const envMap = this.scene.getEnvMap()!;
        const width = Math.pow(2, Math.ceil(Math.log2(envMap.texture.width)));
        const height = Math.pow(2, Math.ceil(Math.log2(envMap.texture.height)));

        this.envLightLuminanceTexture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width,
            height,
            format: ResourceFormat.R32Float,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget,
            name: "RTXDI::envLuminance",
        });
        this.envLightPdfTexture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width,
            height,
            mipLevels: Math.floor(Math.log2(Math.max(width, height))) + 1,
            format: ResourceFormat.R32Float,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget,
            name: "RTXDI::envPdf",
        });

        const root = this.updateEnvLightPass!.getRootVar();
        this.scene.bindShaderData(root);
        const v = root["gEnvLightUpdater"] as ShaderVar;
        v["envLightLuminance"] = this.envLightLuminanceTexture;
        v["envLightPdf"] = this.envLightPdfTexture;
        v["texDim"] = [width, height];
        this.updateEnvLightPass!.execute(ctx, width, height);

        this.generateMips(ctx, this.envLightPdfTexture);
        this.envLightValid = true;
    }

    /** Downsample chain via blits (mirrors Texture::generateMips box filtering). */
    private generateMips(ctx: RenderContext, tex: Texture): void {
        for (let mip = 1; mip < tex.mipCount; mip++) {
            ctx.blit(tex, tex, "linear", mip - 1, mip);
        }
    }

    private presampleLights(ctx: RenderContext, motionVectors: Texture): void {
        {
            const root = this.presampleLocalLightsPass!.getRootVar();
            this.bindShaderDataInternal(root, motionVectors);
            this.presampleLocalLightsPass!.execute(ctx, this.options.presampledTileSize, this.options.presampledTileCount);
        }
        if (this.envLightPresent) {
            const root = this.presampleEnvLightPass!.getRootVar();
            this.bindShaderDataInternal(root, motionVectors);
            this.presampleEnvLightPass!.execute(ctx, this.options.presampledTileSize, this.options.presampledTileCount);
        }
    }

    private testCandidateVisibility(ctx: RenderContext, motionVectors: Texture): void {
        if (!this.options.testCandidateVisibility) return;
        this.dispatchScreen(ctx, this.testCandidateVisibilityPass!, motionVectors, { gOutputReservoirID: kCandidateReservoirID });
    }

    private spatialResampling(ctx: RenderContext, motionVectors: Texture, inputReservoirID: number): number {
        let inputID = inputReservoirID;
        let outputID = inputID !== 1 ? 1 : 0;
        for (let i = 0; i < this.options.spatialIterations; i++) {
            this.dispatchScreen(ctx, this.spatialResamplingPass!, motionVectors, { gInputReservoirID: inputID, gOutputReservoirID: outputID });
            [inputID, outputID] = [outputID, inputID];
        }
        return inputID;
    }

    private dispatchScreen(ctx: RenderContext, pass: ComputePass, motionVectors: Texture, cb: Record<string, number>): void {
        const root = pass.getRootVar();
        const cbVar = root["CB"] as ShaderVar;
        for (const [k, val] of Object.entries(cb)) (cbVar as Record<string, unknown>)[k] = val;
        this.bindShaderDataInternal(root, motionVectors);
        pass.execute(ctx, this.frameDim[0], this.frameDim[1]);
    }

    /** Mirrors RTXDI::bindShaderDataInternal (params set member-wise; ReGIR zeroed). */
    private bindShaderDataInternal(rootVar: ShaderVar, motionVectors: Texture | null): void {
        this.scene.bindShaderData(rootVar);
        const v = rootVar["gRTXDI"] as ShaderVar;

        // RTXDI_ResamplingRuntimeParameters (FillRuntimeParameters; ReGIR disabled -> zeros).
        const p = v["params"] as ShaderVar;
        p["firstLocalLight"] = 0;
        p["numLocalLights"] = this.localLightCount;
        p["firstInfiniteLight"] = this.localLightCount;
        p["numInfiniteLights"] = this.infiniteLightCount;
        p["environmentLightPresent"] = this.envLightPresent ? 1 : 0;
        p["environmentLightIndex"] = this.envLightIndex;
        p["neighborOffsetMask"] = kNeighborOffsetCount - 1;
        p["tileSize"] = this.options.presampledTileSize;
        p["tileCount"] = this.options.presampledTileCount;
        p["enableLocalLightImportanceSampling"] = 1;
        p["reservoirBlockRowPitch"] = this.reservoirBlockRowPitch;
        p["reservoirArrayPitch"] = this.reservoirArrayPitch;
        p["environmentRisBufferOffset"] = this.options.presampledTileCount * this.options.presampledTileSize;
        p["environmentTileCount"] = this.options.presampledTileCount;
        p["environmentTileSize"] = this.options.presampledTileSize;
        p["uniformRandomNumber"] = jenkinsHash(this.frameIndex);
        p["activeCheckerboardField"] = 0;

        v["frameIndex"] = this.frameIndex;
        v["rayEpsilon"] = this.options.rayEpsilon;
        v["frameDim"] = this.frameDim;
        v["pixelCount"] = this.frameDim[0] * this.frameDim[1];
        v["storeCompactLightInfo"] = this.options.storeCompactLightInfo ? 1 : 0;
        v["useEmissiveTextures"] = this.options.useEmissiveTextures ? 1 : 0;
        v["currentSurfaceBufferIndex"] = this.currentSurfaceBufferIndex;
        v["prevSurfaceBufferIndex"] = 1 - this.currentSurfaceBufferIndex;

        v["localLightCandidateCount"] = this.options.localLightCandidateCount;
        v["infiniteLightCandidateCount"] = this.options.infiniteLightCandidateCount;
        v["envLightCandidateCount"] = this.options.envLightCandidateCount;
        v["brdfCandidateCount"] = this.options.brdfCandidateCount;

        v["maxHistoryLength"] = this.options.maxHistoryLength;
        v["biasCorrectionMode"] = this.options.biasCorrection;
        v["finalShadingReservoir"] = this.lastFrameReservoirID;

        v["spatialSampleCount"] = this.options.spatialSampleCount;
        v["disocclusionSampleCount"] = this.options.spatialSampleCount;
        v["samplingRadius"] = this.options.samplingRadius;
        v["depthThreshold"] = this.options.depthThreshold;
        v["normalThreshold"] = this.options.normalThreshold;
        v["boilingFilterStrength"] = this.options.boilingFilterStrength;
        v["enableVisibilityShortcut"] = this.options.enableVisibilityShortcut ? 1 : 0;
        v["enablePermutationSampling"] = this.options.enablePermutationSampling ? 1 : 0;

        const prev = this.prevCameraData ?? this.scene.camera.getData();
        v["prevCameraU"] = prev.cameraU.toArray();
        v["prevCameraV"] = prev.cameraV.toArray();
        v["prevCameraW"] = prev.cameraW.toArray();
        v["prevCameraJitter"] = [prev.jitterX, prev.jitterY];

        v["lightInfo"] = this.lightInfoBuffer!;
        v["surfaceData"] = this.surfaceDataBuffer!;
        v["risBuffer"] = this.lightTileBuffer!;
        v["compactLightInfo"] = this.compactLightInfoBuffer!;
        v["reservoirs"] = this.reservoirBuffer!;
        v["neighborOffsets"] = this.neighborOffsetsBuffer!;
        if (motionVectors) v["motionVectors"] = motionVectors;

        v["localLightPdfTexture"] = this.localLightPdfTexture!;
        if (this.envLightLuminanceTexture) v["envLightLuminanceTexture"] = this.envLightLuminanceTexture;
        if (this.envLightPdfTexture) v["envLightPdfTexture"] = this.envLightPdfTexture;
    }
}
