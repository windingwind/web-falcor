/**
 * Mirrors Scene/Material/MaterialParamLayout + SerializedMaterialParams and the per-material
 * layouts (StandardMaterialParamLayout, PBRTDiffuse/PBRTConductorMaterialParamLayout.slang):
 * a material's differentiable parameters as 20 floats, the vector BSDFOptimizer and the
 * scene's get/set_material_params exchange. Values go through float16 like native's getters,
 * which read the packed BasicMaterialData.
 */

import { float3, float4 } from "../../Utils/Math/Vector.js";
import { RuntimeError } from "../../Core/Error.js";
import { MaterialType, ShadingModel, type BasicMaterialDesc } from "./MaterialData.js";
import { float16ToFloat32, float32ToFloat16 } from "../../Utils/Math/Float16.js";
import type { SceneMaterialDesc } from "../Scene.js";

/** Mirrors SerializedMaterialParams::kParamCount. */
export const kMaterialParamCount = 20;

/** Mirrors MaterialParamLayoutEntry. */
export interface MaterialParamLayoutEntry {
    name: string;
    pythonName: string;
    size: number;
    offset: number;
}

const kEpsilon = 1e-4;
const f16 = (v: number) => float16ToFloat32(float32ToFloat16(v));

interface ParamDef extends MaterialParamLayoutEntry {
    get(m: SceneMaterialDesc): number[];
    set(m: SceneMaterialDesc, v: number[]): void;
    /** detail::clampMaterialParam bounds (default [1e-4, 1 - 1e-4]). */
    min?: number | number[];
    max?: number;
}

const basic = (m: SceneMaterialDesc): BasicMaterialDesc => m.basic;
const baseColor3: Pick<ParamDef, "get" | "set"> = {
    get: (m) => {
        const c = basic(m).baseColor ?? new float4(1, 1, 1, 1);
        return [c.x, c.y, c.z].map(f16);
    },
    set: (m, v) => {
        const a = basic(m).baseColor?.w ?? 1;
        basic(m).baseColor = new float4(v[0]!, v[1]!, v[2]!, a);
    },
};
const transmission: Pick<ParamDef, "get" | "set"> = {
    get: (m) => {
        const t = basic(m).transmission ?? new float3(1, 1, 1);
        return [t.x, t.y, t.z].map(f16);
    },
    set: (m, v) => void (basic(m).transmission = new float3(v[0]!, v[1]!, v[2]!)),
};
const specular = (m: SceneMaterialDesc) => basic(m).specular ?? new float4(0, 0, 0, 0);
const setSpecular = (m: SceneMaterialDesc, i: number, v: number) => {
    const s = specular(m);
    const c = [s.x, s.y, s.z, s.w];
    c[i] = v;
    basic(m).specular = new float4(c[0]!, c[1]!, c[2]!, c[3]!);
};
const entry = (name: string, pythonName: string, size: number, offset: number, rest: Pick<ParamDef, "get" | "set"> & Partial<ParamDef>): ParamDef => ({ name, pythonName, size, offset, ...rest });

const kStandard: ParamDef[] = [
    entry("baseColor", "base_color", 3, 0, baseColor3),
    entry("metallic", "metallic", 1, 3, { get: (m) => [f16(specular(m).z)], set: (m, v) => setSpecular(m, 2, v[0]!) }),
    entry("roughness", "roughness", 1, 4, { get: (m) => [f16(specular(m).y)], set: (m, v) => setSpecular(m, 1, v[0]!), min: 0.05 }),
    entry("ior", "ior", 1, 5, {
        get: (m) => [f16(m.header?.ior ?? 1.5)],
        set: (m, v) => void (m.header = { ...m.header, ior: v[0]! }),
        min: 1e-3,
        max: 1e3,
    }),
    entry("transmissionColor", "transmission_color", 3, 6, transmission),
    entry("diffuseTransmission", "diffuse_transmission", 1, 9, { get: (m) => [f16(basic(m).diffuseTransmission ?? 0)], set: (m, v) => void (basic(m).diffuseTransmission = v[0]!), min: 0 }),
    entry("specularTransmission", "specular_transmission", 1, 10, { get: (m) => [f16(basic(m).specularTransmission ?? 0)], set: (m, v) => void (basic(m).specularTransmission = v[0]!), min: 0 }),
    entry("emissiveColor", "emissive_color", 3, 11, {
        get: (m) => {
            const e = basic(m).emissive ?? new float3(0, 0, 0);
            return [e.x, e.y, e.z];
        },
        set: (m, v) => void (basic(m).emissive = new float3(v[0]!, v[1]!, v[2]!)),
    }),
    entry("emissiveFactor", "emissive_factor", 1, 14, { get: (m) => [basic(m).emissiveFactor ?? 1], set: (m, v) => void (basic(m).emissiveFactor = v[0]!), min: 1e-8, max: 1e8 }),
];

const kPBRTDiffuse: ParamDef[] = [entry("baseColor", "diffuse", 3, 0, baseColor3)];

const kPBRTConductor: ParamDef[] = [
    entry("baseColor", "eta", 3, 0, baseColor3),
    entry("transmissionColor", "k", 3, 3, transmission),
    entry("roughness", "roughness", 2, 6, {
        get: (m) => [f16(specular(m).x), f16(specular(m).y)],
        set: (m, v) => {
            setSpecular(m, 0, v[0]!);
            setSpecular(m, 1, v[1]!);
        },
        min: 0.05,
    }),
];

function layoutOf(m: SceneMaterialDesc): ParamDef[] {
    const type = m.header?.materialType ?? MaterialType.Standard;
    if (type === MaterialType.Standard) {
        if ((m.basic.shadingModel ?? ShadingModel.MetalRough) !== ShadingModel.MetalRough) throw new RuntimeError("Only MetalRough shading model is supported in parameter layout.");
        return kStandard;
    }
    if (type === MaterialType.PBRTDiffuse) return kPBRTDiffuse;
    if (type === MaterialType.PBRTConductor) return kPBRTConductor;
    throw new RuntimeError("Material does not have a parameter layout.");
}

/** Mirrors Material::getParamLayout. */
export function getMaterialParamLayout(m: SceneMaterialDesc): MaterialParamLayoutEntry[] {
    return layoutOf(m).map(({ name, pythonName, size, offset }) => ({ name, pythonName, size, offset }));
}

/** Mirrors Material::serializeParams. */
export function serializeMaterialParams(m: SceneMaterialDesc): Float32Array {
    const params = new Float32Array(kMaterialParamCount);
    for (const p of layoutOf(m)) params.set(p.get(m), p.offset);
    return params;
}

/** Mirrors Material::deserializeParams (clamped like detail::clampMaterialParam). */
export function deserializeMaterialParams(m: SceneMaterialDesc, params: ArrayLike<number>, clamp = true): void {
    for (const p of layoutOf(m)) {
        const v = Array.from({ length: p.size }, (_, i) => params[p.offset + i]!);
        const lo = p.min ?? kEpsilon;
        const hi = p.max ?? 1 - kEpsilon;
        p.set(m, clamp ? v.map((x, i) => Math.min(Math.max(x, Array.isArray(lo) ? lo[i]! : lo), hi)) : v);
    }
}
