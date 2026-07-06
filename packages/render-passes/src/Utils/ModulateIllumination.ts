/**
 * Illumination compositing pass mirroring Source/RenderPasses/ModulateIllumination.
 * All inputs optional; is_valid_* defines come from connected textures ANDed
 * with the use* flags (native getValidResourceDefines semantics).
 */

import {
    ComputePass,
    FieldFlags,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/ModulateIllumination/ModulateIllumination.cs.slang";

/** [input name, shader texture name, use-flag property] (kInputChannels). */
const kChannels: [string, string, string][] = [
    ["emission", "gEmission", "useEmission"],
    ["diffuseReflectance", "gDiffuseReflectance", "useDiffuseReflectance"],
    ["diffuseRadiance", "gDiffuseRadiance", "useDiffuseRadiance"],
    ["specularReflectance", "gSpecularReflectance", "useSpecularReflectance"],
    ["specularRadiance", "gSpecularRadiance", "useSpecularRadiance"],
    ["deltaReflectionEmission", "gDeltaReflectionEmission", "useDeltaReflectionEmission"],
    ["deltaReflectionReflectance", "gDeltaReflectionReflectance", "useDeltaReflectionReflectance"],
    ["deltaReflectionRadiance", "gDeltaReflectionRadiance", "useDeltaReflectionRadiance"],
    ["deltaTransmissionEmission", "gDeltaTransmissionEmission", "useDeltaTransmissionEmission"],
    ["deltaTransmissionReflectance", "gDeltaTransmissionReflectance", "useDeltaTransmissionReflectance"],
    ["deltaTransmissionRadiance", "gDeltaTransmissionRadiance", "useDeltaTransmissionRadiance"],
    ["residualRadiance", "gResidualRadiance", "useResidualRadiance"],
];

export class ModulateIllumination extends RenderPass {
    private use = new Map<string, boolean>();
    private pass: ComputePass | null = null;
    private passKey = "";

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [, , flag] of kChannels) this.use.set(flag, props.get(flag, true));
    }

    override getProperties(): Properties {
        const props = new Properties();
        for (const [, , flag] of kChannels) props.set(flag, this.use.get(flag)!);
        return props;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        for (const [name] of kChannels) {
            r.addInput(name, name).bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        }
        const [w, h] = compileData.defaultTexDims;
        r.addOutput("output", "output")
            .bindFlags(ResourceBindFlags.UnorderedAccess)
            .format(ResourceFormat.RGBA32Float)
            .texture2D(w, h);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const output = renderData.getTexture("output")!;

        const defines: Record<string, number> = {};
        for (const [name, texname, flag] of kChannels) {
            defines[`is_valid_${texname}`] = renderData.getTexture(name) && this.use.get(flag) ? 1 : 0;
        }
        const key = JSON.stringify(defines);
        if (!this.pass || this.passKey !== key) {
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.passKey = key;
        }

        const root = this.pass.getRootVar();
        root["CB"]["frameDim"] = [output.width, output.height];
        for (const [name, texname, flag] of kChannels) {
            const tex = renderData.getTexture(name);
            if (tex && this.use.get(flag)) root[texname] = tex;
        }
        root["gOutput"] = output;
        this.pass.execute(ctx, output.width, output.height);
    }
}

registerRenderPass("ModulateIllumination", (device, props) => new ModulateIllumination(device, props));
