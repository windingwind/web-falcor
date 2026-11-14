/**
 * Host side of the visibility service (Visibility.slangh): picks VIS_MODE and binds the shadow-map data, so any compute
 * pass can query light visibility at arbitrary world positions.
 *
 * Usage in a pass:
 *   reflect():  VisibilityService.addInputs(r)                         // optional "shadowMap" input (CascadedShadowMap.shadowMap)
 *   execute():  const vis = VisibilityService.resolve(mode, renderData, scene, lightIndex);
 *               defines.addAll(vis.defines)  ->  create the ComputePass  ->  vis.bind(rootVar, params)
 *   shader:     #include "RenderPasses/RasterLighting/Visibility.slangh"  ...  visibility(posW, faceN, L, dist)
 * Modes: "None", "ShadowMap" (needs the CascadedShadowMap output and its per-frame data; falls back to "None" without them,
 * like DeferredLighting), "RayTraced" (software BVH, needs a scene).
 */

import { FieldFlags, ResourceBindFlags, type RenderData, type RenderPassReflection, type Scene, type ShaderVar } from "@web-falcor/falcor";
import { kCascadedShadowMapData, kMaxCascades, type CascadedShadowMapData } from "./CascadedShadowMap.js";

export const kVisibilityModes = ["None", "ShadowMap", "RayTraced"] as const;
export type VisibilityMode = (typeof kVisibilityModes)[number];

export interface VisibilityParams {
    /** Constant shadow depth bias in world units (DeferredLighting default 0.002). */
    depthBias: number;
    /** Normal offset in shadow texels (DeferredLighting default 1.5). */
    normalBias: number;
    /** PCF kernel radius in texels (DeferredLighting default 1). */
    pcfRadius: number;
}

export const kDefaultVisibilityParams: VisibilityParams = { depthBias: 0.002, normalBias: 1.5, pcfRadius: 1 };

export interface ResolvedVisibility {
    /** The mode actually used (ShadowMap without a shadow map resolves to None). */
    mode: VisibilityMode;
    /** VIS_MODE define (0/1/2). */
    defines: Record<string, number>;
    /** The analytic light the visibility belongs to: the shadow-map light, else `lightIndex` (default: first directional). */
    lightIndex: number;
    /** Binds VisibilityCB / gVisShadowMap (ShadowMap mode; no-op otherwise). */
    bind(root: ShaderVar, params?: Partial<VisibilityParams>): void;
}

export class VisibilityService {
    /** Adds the optional shadow-map input (connect CascadedShadowMap.shadowMap for the ShadowMap mode). */
    static addInputs(r: RenderPassReflection, name = "shadowMap"): void {
        r.addInput(name, "CascadedShadowMap.shadowMap (ShadowMap visibility)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
    }

    static parseMode(pass: string, v: string): VisibilityMode {
        if (!(kVisibilityModes as readonly string[]).includes(v)) throw new Error(`${pass}: unknown visibility mode '${v}' (${kVisibilityModes.join(", ")})`);
        return v as VisibilityMode;
    }

    static resolve(mode: VisibilityMode, renderData: RenderData, scene: Scene | null, lightIndex = -1, inputName = "shadowMap"): ResolvedVisibility {
        const csm = renderData.dictionary.get(kCascadedShadowMapData) as CascadedShadowMapData | undefined;
        const shadowMap = renderData.getTexture(inputName);
        const lights = scene?.activeLights ?? [];
        const firstDirectional = lights.findIndex((l) => l.type === 1);
        const used: VisibilityMode = mode === "ShadowMap" ? (shadowMap && csm && scene ? "ShadowMap" : "None") : mode === "RayTraced" && !scene ? "None" : mode;
        const light = used === "ShadowMap" ? csm!.lightIndex : lightIndex >= 0 ? lightIndex : firstDirectional;
        return {
            mode: used,
            defines: { VIS_MODE: used === "ShadowMap" ? 1 : used === "RayTraced" ? 2 : 0 },
            lightIndex: light,
            bind(root: ShaderVar, params: Partial<VisibilityParams> = {}): void {
                if (used !== "ShadowMap") return;
                const p = { ...kDefaultVisibilityParams, ...params };
                const cam = scene!.camera;
                const pos = cam.getPosition(), tgt = cam.getTarget();
                const fwd = [tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z];
                const len = Math.hypot(fwd[0]!, fwd[1]!, fwd[2]!);
                const pad = (a: number[]) => [0, 1, 2, 3].map((i) => a[Math.min(i, a.length - 1)]!);
                const cb = root["VisibilityCB"]!;
                cb["gVisCascadeCount"] = csm!.cascadeCount;
                cb["gVisCascadeSize"] = csm!.size;
                cb["gVisPcfRadius"] = p.pcfRadius;
                cb["gVisDepthBias"] = p.depthBias;
                cb["gVisCascadeViewProj"] = Array.from({ length: kMaxCascades }, (_v, c) => csm!.viewProj[Math.min(c, csm!.cascadeCount - 1)]!);
                cb["gVisCascadeSplits"] = pad(csm!.splits);
                cb["gVisCascadeTexelWorld"] = pad(csm!.texelWorld);
                cb["gVisCascadeDepthScale"] = pad(csm!.depthRange.map((r) => 1 / r));
                cb["gVisCamForward"] = fwd.map((x) => x / len);
                cb["gVisNormalBias"] = p.normalBias;
                root["gVisShadowMap"] = shadowMap!;
            },
        };
    }
}
