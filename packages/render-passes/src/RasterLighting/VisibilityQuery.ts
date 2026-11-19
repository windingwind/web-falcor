/**
 * VisibilityQuery: per-pixel visibility of one analytic light at the visible surface, through the shared visibility
 * service (Visibility.slangh / VisibilityService.ts). Doubles as an AOV source (shadow mask) and as the reference user of
 * the service.
 *
 * Inputs : vbuffer (GBufferRaster/VBuffer), shadowMap (optional: CascadedShadowMap.shadowMap, for mode ShadowMap).
 * Outputs: visibility (RGBA32Float): x = visibility, y = dot(oriented face normal, L), z = distance to the light,
 *          w = 1 on geometry with a valid light sample.
 * Params : mode (None | ShadowMap | RayTraced), lightIndex (-1 = the shadow-map light in ShadowMap mode, else the first
 *          directional light), depthBias, normalBias, pcfRadius (ShadowMap mode; DeferredLighting's defaults).
 */

import {
    ComputePass,
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
    type UIWidgets,
} from "@web-falcor/falcor";
import { VisibilityService, kDefaultVisibilityParams, kVisibilityModes, type VisibilityMode } from "./VisibilityService.js";

const kShaderFile = "RenderPasses/RasterLighting/VisibilityQuery.cs.slang";

/** Queries the shared shadow-map or ray-traced visibility service. */
export class VisibilityQuery extends RenderPass {
    private mPass: ComputePass | null = null;
    private mPassKey: string = "";
    private mMode: VisibilityMode = "ShadowMap";
    private mParameters: { depthBias: number; normalBias: number; pcfRadius: number; lightIndex: number } = {
        lightIndex: -1,
        ...kDefaultVisibilityParams,
    };

    constructor(device: Device, props: Properties) {
        super(device);
        this.mMode = VisibilityService.parseMode("VisibilityQuery", props.get<string>("mode", "ShadowMap"));
        for (const k of Object.keys(this.mParameters) as (keyof typeof this.mParameters)[])
            this.mParameters[k] = Number(props.get(k, this.mParameters[k]));
    }

    override getProperties(): Properties {
        return new Properties({ mode: this.mMode, ...this.mParameters });
    }

    override renderUI(ui: UIWidgets): void {
        ui.dropdown("Mode", [...kVisibilityModes], this.mMode, (v) => (this.mMode = v as VisibilityMode));
        ui.slider("Depth bias (world)", this.mParameters.depthBias, 0, 0.1, 0.0005, (v) => (this.mParameters.depthBias = v));
        ui.slider("Normal bias (texels)", this.mParameters.normalBias, 0, 8, 0.1, (v) => (this.mParameters.normalBias = v));
        ui.slider("PCF radius", this.mParameters.pcfRadius, 0, 4, 1, (v) => (this.mParameters.pcfRadius = Math.round(v)));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Packed hit information (V-buffer)").bindFlags(ResourceBindFlags.ShaderResource);
        VisibilityService.addInputs(r);
        r.addOutput("visibility", "x = visibility, y = N.L, z = light distance, w = valid")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.mPass = null;
        this.mPassKey = "";
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const out = renderData.getTexture("visibility")!;
        if (!this.scene) {
            ctx.clearTexture(out, [1, 0, 0, 0]);
            return;
        }
        const vis = VisibilityService.resolve(this.mMode, renderData, this.scene, this.mParameters.lightIndex);
        const defines = this.scene.getSceneDefines().addAll(vis.defines);
        const key = defines.key();
        if (!this.mPass || key !== this.mPassKey) {
            this.mPass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.mPassKey = key;
        }
        const root = this.mPass.getRootVar();
        this.scene.bindShaderData(root);
        root["CB"]!["gFrameDim"] = [out.width, out.height];
        root["CB"]!["gLightIndex"] = vis.lightIndex;
        vis.bind(root, this.mParameters);
        root["gVBuffer"] = renderData.getTexture("vbuffer")!;
        root["gOutput"] = out;
        this.mPass.execute(ctx, out.width, out.height);
    }
}

registerRenderPass("VisibilityQuery", (device, props) => new VisibilityQuery(device, props), {
    desc: "Per-pixel light visibility through the shared visibility service (shadow map or BVH rays)",
});
