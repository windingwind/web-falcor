/**
 * Mirrors RenderPasses/BSDFOptimizer: fits one material's BSDF parameters to another's by
 * differentiating the L2 loss between their BSDF slices (backward-mode autodiff through the
 * material's evalAD) and stepping Adam; the viewer shows initial | |difference| | reference.
 * The Python surface matches native (init_material_id, ref_material_id, bsdf_slice_resolution,
 * compute_bsdf_grads()).
 *
 * §9: the gradient readback is asynchronous (native blocks on a fence), so an optimization step
 * lands a frame or more after its gradients were computed; the optimizer kernel compiles with
 * the pinned pre-refactor slang-wasm, like WARDiffPathTracer's backward modes.
 */

import {
    Buffer,
    ComputePass,
    GradientType,
    MaterialType,
    Properties,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    SAMPLE_GENERATOR_UNIFORM,
    SampleGenerator,
    SceneGradients,
    deserializeMaterialParams,
    getMaterialParamLayout,
    kMaterialParamCount,
    registerRenderPass,
    serializeMaterialParams,
    type CompileData,
    type Device,
    type RenderContext,
    type RenderData,
    type Scene,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";

const kOptimizerPass = "RenderPasses/BSDFOptimizer/BSDFOptimizer.cs.slang";
const kViewerPass = "RenderPasses/BSDFOptimizer/BSDFViewer.cs.slang";
const kAutodiffSlang = "/tools/slang-wasm-2026.5.2/slang-wasm.js";

/** Native kLearningRates, by material type and Python parameter name. */
const kLearningRates: Partial<Record<MaterialType, Record<string, number>>> = {
    [MaterialType.PBRTDiffuse]: { diffuse: 1e-2 },
    [MaterialType.PBRTConductor]: { eta: 1e-2, k: 1e-2, roughness: 1e-2 },
    [MaterialType.Standard]: { base_color: 1e-2, roughness: 3e-3, metallic: 3e-3 },
};

/** Mirrors BSDFOptimizer::AdamOptimizer. */
class AdamOptimizer {
    private m: number[] = [];
    private v: number[] = [];
    private steps = 0;

    constructor(
        private readonly lr: number[],
        private readonly beta1 = 0.9,
        private readonly beta2 = 0.999,
        private readonly epsilon = 1e-6,
    ) {}

    step(dx: ArrayLike<number>, x: Float32Array): void {
        if (this.lr.length !== dx.length || this.lr.length !== x.length) throw new Error("AdamOptimizer::step(): lr, dx, and x must have the same size.");
        if (this.m.length === 0) {
            this.m = new Array<number>(dx.length).fill(0);
            this.v = new Array<number>(dx.length).fill(0);
        }
        this.steps++;
        for (let i = 0; i < dx.length; i++) {
            if (this.lr[i] === 0) continue;
            const g = dx[i]!;
            // Single-precision arithmetic like native's floats.
            this.m[i] = Math.fround(this.beta1 * this.m[i]! + (1 - this.beta1) * g);
            this.v[i] = Math.fround(this.beta2 * this.v[i]! + (1 - this.beta2) * g * g);
            const mHat = this.m[i]! / (1 - Math.pow(this.beta1, this.steps));
            const vHat = this.v[i]! / (1 - Math.pow(this.beta2, this.steps));
            x[i] = Math.fround(x[i]! - (this.lr[i]! * mHat) / (Math.sqrt(vHat) + this.epsilon));
        }
    }
}

export class BSDFOptimizer extends RenderPass {
    private params = {
        frameDim: [0, 0] as [number, number],
        frameCount: 0,
        initViewPortOffset: [0, 0] as [number, number],
        diffViewPortOffset: [0, 0] as [number, number],
        refViewPortOffset: [0, 0] as [number, number],
        viewPortScale: [0, 0] as [number, number],
        bsdfTableDim: [0, 0] as [number, number],
        initMaterialID: 0,
        refMaterialID: 1,
    };
    private readonly sampleGenerator: SampleGenerator;
    private sceneGradients: SceneGradients | null = null;
    private optimizerPass: ComputePass | null = null;
    private viewerPass: ComputePass | null = null;
    private initParams: Float32Array = new Float32Array(kMaterialParamCount);
    private refParams: Float32Array = new Float32Array(kMaterialParamCount);
    private curParams: Float32Array = new Float32Array(kMaterialParamCount);
    private adam = new AdamOptimizer(new Array<number>(kMaterialParamCount).fill(0));
    /** Mirrors mRunOptimization (the Start/Stop buttons). */
    runOptimization = false;
    private stepPending = false;
    /** Optimization steps taken since the last (re)initialization. */
    stepCount = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key, value] of props.entries()) {
            if (key === "initMaterialID") this.params.initMaterialID = Number(value);
            else if (key === "refMaterialID") this.params.refMaterialID = Number(value);
            else console.warn(`Unknown property '${key}' in BSDFOptimizer properties.`);
        }
        this.sampleGenerator = SampleGenerator.create(device, SAMPLE_GENERATOR_UNIFORM);
    }

    override getProperties(): Properties {
        return new Properties({ initMaterialID: this.params.initMaterialID, refMaterialID: this.params.refMaterialID });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("output", "Output buffer").format(ResourceFormat.RGBA32Float).bindFlags(ResourceBindFlags.UnorderedAccess);
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        const [w, h] = compileData.defaultTexDims;
        this.params.frameDim = [w, h];
        // Viewports: left the initial material, middle the absolute difference, right the reference.
        const extent = Math.min(Math.floor(w / 3), h);
        this.params.bsdfTableDim = [extent, extent];
        const xOffset = Math.floor((w - extent * 3) / 2);
        const yOffset = Math.floor((h - extent) / 2);
        this.params.initViewPortOffset = [xOffset, yOffset];
        this.params.diffViewPortOffset = [xOffset + extent, yOffset];
        this.params.refViewPortOffset = [xOffset + extent * 2, yOffset];
        this.params.viewPortScale = [1 / extent, 1 / extent];
    }

    override async initAsync(): Promise<void> {
        await this.device.programManager.loadSlangRuntime(kAutodiffSlang);
    }

    // --- Python surface (native registerBindings) ---
    get init_material_id(): number {
        return this.params.initMaterialID;
    }
    get ref_material_id(): number {
        return this.params.refMaterialID;
    }
    get bsdf_slice_resolution(): number {
        return this.params.bsdfTableDim[0];
    }
    set bsdf_slice_resolution(reso: number) {
        this.params.bsdfTableDim = [reso, reso];
        this.params.viewPortScale = [1 / reso, 1 / reso];
    }
    /** Mirrors computeBSDFGrads: runs the optimizer kernel and returns the aggregated gradients. */
    compute_bsdf_grads(): Buffer {
        this.executeOptimizerPass(this.device.renderContext);
        return this.sceneGradients!.getGradsBuffer(GradientType.Material)!;
    }

    get currentParams(): Float32Array {
        return this.curParams;
    }
    get referenceParams(): Float32Array {
        return this.refParams;
    }

    override setScene(scene: Scene | null): void {
        super.setScene(scene);
        this.optimizerPass = null;
        this.viewerPass = null;
        if (!scene) return;
        if (this.params.bsdfTableDim[0] === 0) this.bsdf_slice_resolution = 128;
        const defines = scene.getSceneDefines().addAll(this.sampleGenerator.getDefines());
        this.viewerPass = ComputePass.create(this.device, { path: kViewerPass, defines });
        this.sceneGradients = new SceneGradients(this.device, [{ type: GradientType.Material, dim: kMaterialParamCount, hashSize: 64 }]);
        this.initParams = serializeMaterialParams(scene.getMaterial(this.params.initMaterialID));
        this.refParams = serializeMaterialParams(scene.getMaterial(this.params.refMaterialID));
        this.initOptimization();
    }

    /** Mirrors initOptimization: resets the initial material and the Adam state. */
    initOptimization(): void {
        const scene = this.scene!;
        const material = scene.getMaterial(this.params.initMaterialID);
        deserializeMaterialParams(material, this.initParams);
        scene.updateMaterial(this.params.initMaterialID);
        this.params.frameCount = 0;
        this.curParams = this.initParams.slice();
        this.stepCount = 0;
        const lr = new Array<number>(kMaterialParamCount).fill(0);
        const rates = kLearningRates[material.header?.materialType ?? MaterialType.Standard];
        if (rates) {
            for (const p of getMaterialParamLayout(material)) {
                const rate = rates[p.pythonName];
                if (rate !== undefined) for (let i = 0; i < p.size; i++) lr[p.offset + i] = rate;
            }
        }
        this.adam = new AdamOptimizer(lr);
    }

    private bindParams(v: ShaderVar): void {
        const p = this.params;
        for (const [k, value] of Object.entries(p)) v[k] = value;
    }

    private executeOptimizerPass(ctx: RenderContext): void {
        if (!this.scene || !this.sceneGradients) return;
        // Created on first use: setScene can precede initAsync, which loads the pinned Slang.
        this.optimizerPass ??= ComputePass.create(this.device, {
            path: kOptimizerPass,
            defines: this.scene.getSceneDefines().addAll(this.sampleGenerator.getDefines()),
            slangRuntime: kAutodiffSlang,
        });
        this.sceneGradients.clearGrads(ctx, GradientType.Material);
        const root = this.optimizerPass.getRootVar();
        this.bindParams((root["CB"] as ShaderVar)["params"] as ShaderVar);
        this.sceneGradients.bindShaderData(root["gSceneGradients"] as ShaderVar);
        this.scene.bindShaderData(root);
        this.optimizerPass.execute(ctx, this.params.bsdfTableDim[0], this.params.bsdfTableDim[1], 1);
        this.sceneGradients.aggregateGrads(ctx, GradientType.Material);
    }

    /** Mirrors step: Adam over the read-back gradients, then the material update. */
    private async step(ctx: RenderContext): Promise<void> {
        const bytes = await ctx.readBuffer(this.sceneGradients!.getGradsBuffer(GradientType.Material)!);
        const grads = new Float32Array(bytes.buffer, bytes.byteOffset, kMaterialParamCount);
        this.adam.step(grads, this.curParams);
        deserializeMaterialParams(this.scene!.getMaterial(this.params.initMaterialID), this.curParams);
        this.scene!.updateMaterial(this.params.initMaterialID);
        this.stepCount++;
        // Stop once the mean relative L1 parameter error is below 1e-3.
        let relL1 = 0;
        for (let i = 0; i < kMaterialParamCount; i++) relL1 += Math.abs(this.curParams[i]! - this.refParams[i]!) / Math.max(this.refParams[i]!, 1e-6);
        if (relL1 / kMaterialParamCount < 1e-3) this.runOptimization = false;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const output = renderData.getTexture("output")!;
        if (this.runOptimization && !this.stepPending) {
            this.executeOptimizerPass(ctx);
            this.stepPending = true;
            void this.step(ctx).finally(() => (this.stepPending = false));
        }
        if (!this.scene || !this.viewerPass || this.scene.getMaterialCount() === 0) {
            ctx.clearTexture(output);
        } else {
            const root = this.viewerPass.getRootVar();
            const v = root["gBSDFViewer"] as ShaderVar;
            this.bindParams(v["params"] as ShaderVar);
            v["output"] = output;
            this.scene.bindShaderData(root);
            this.viewerPass.execute(ctx, this.params.frameDim[0], this.params.frameDim[1], 1);
        }
        this.params.frameCount++;
    }

    override renderUI(ui: UIWidgets): void {
        if (!this.scene || this.scene.getMaterialCount() === 0) {
            ui.text("No scene/materials loaded.");
            return;
        }
        ui.button("Start optimization", () => {
            if (this.params.frameCount > 0) this.initOptimization();
            this.runOptimization = true;
        });
        ui.button("Stop optimization", () => (this.runOptimization = false));
        ui.button("Reset optimization", () => {
            this.initOptimization();
            this.runOptimization = false;
        });
        const names = Array.from({ length: this.scene.getMaterialCount() }, (_, i) => `${i}: ${this.scene!.getMaterial(i).name ?? ""}`);
        ui.dropdown("Initial material", names, names[this.params.initMaterialID]!, (v: string) => {
            this.params.initMaterialID = names.indexOf(v);
            this.initParams = serializeMaterialParams(this.scene!.getMaterial(this.params.initMaterialID));
        });
        ui.dropdown("Reference material", names, names[this.params.refMaterialID]!, (v: string) => {
            this.params.refMaterialID = names.indexOf(v);
            this.refParams = serializeMaterialParams(this.scene!.getMaterial(this.params.refMaterialID));
        });
        ui.text(`Steps: ${this.stepCount}; params: ${Array.from(this.curParams.slice(0, 8), (x) => x.toFixed(3)).join(", ")}`);
    }
}

registerRenderPass("BSDFOptimizer", (device, props) => new BSDFOptimizer(device, props));
