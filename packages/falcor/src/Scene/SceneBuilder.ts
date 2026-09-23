/**
 * SceneBuilder bridge for .pyscene execution (docs §11.1). Builder calls record
 * commands synchronously; resolve() then fetches assets and constructs the Scene.
 */

import type { Device } from "../Core/API/Device.js";
import { Grid } from "./Volume/Grid.js";
import { GridVolume, type GridSlot } from "./Volume/GridVolume.js";
import { buildNanoVDBGrid, type ParsedFloatGrid } from "./Volume/VDBLoader.js";
import { buildSDFGridFromRecipe, type SDFGridRecipe, type SDFGridType } from "./SDFs/SDFGridRecipe.js";
import type { SceneSDFGridDesc } from "./Scene.js";
import { Scene, type SceneMaterialDesc, type SceneMeshDesc } from "./Scene.js";
import type { SceneNode, AnimationChannel, WeightTrack } from "./Animation/SceneAnimation.js";
import { GltfImporter } from "./Importer/GltfImporter.js";
import { FbxImporter, kAssimpSceneExtensions, objMaterialLibraries } from "./Importer/FbxImporter.js";
import { UsdImporter } from "./Importer/UsdImporter.js";
import { convertToLinearSweptSphere, convertToPolytube, extractBasisCurvesFromUsda } from "./Curves/CurveTessellation.js";
import { TextureManager } from "./Material/TextureManager.js";
import { EnvMap } from "./Lights/EnvMap.js";
import { generateTangents } from "./TangentSpace.js";
import { LightType, type AnalyticLight, type StaticVertex } from "./SceneData.js";
import { MaterialType, ShadingModel, packTextureHandle, TextureHandleMode } from "./Material/MaterialData.js";
import { getTextureSlotSrgb } from "./Material/TextureSlots.js";
import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { float4x4, matrixFromTranslation, matrixFromScaling, mulMat } from "../Utils/Math/Matrix.js";
import { RuntimeError } from "../Core/Error.js";
import { AssetCategory, AssetResolver, resolveAssetUrl } from "../Core/AssetResolver.js";
import { Logger } from "../Utils/Logger.js";

/** The python prelude wraps bridge objects in a setattr guard; JS entry
 *  points unwrap back to the underlying bridge instance. */
function unwrapGuard<T>(obj: T): T {
    const inner = (obj as { _o?: T })._o;
    return inner ?? obj;
}

/** TriangleMesh geometry in local space (mirrors Falcor::TriangleMesh). */
export interface TriangleMeshDesc {
    vertices: StaticVertex[];
    indices: Uint32Array;
    /** TriangleMesh.createFromFile: geometry is loaded from this asset in resolve(). */
    _fromFile?: { path: string; smoothNormals: boolean };
}

/** Mirrors TriangleMesh factories (TriangleMesh.cpp). */
export const TriangleMesh = {
    /** Mirrors TriangleMesh::createFromFile; the asset is fetched in resolve(). */
    createFromFile(path: string, smoothNormals = false): TriangleMeshDesc {
        return { vertices: [], indices: new Uint32Array(0), _fromFile: { path: String(path), smoothNormals: !!smoothNormals } };
    },
    createQuad(size: float2 = new float2(1, 1)): TriangleMeshDesc {
        const hx = 0.5 * size.x;
        const hy = 0.5 * size.y;
        const n = new float3(0, 1, 0);
        const t0 = new float4(0, 0, 0, 0);
        const vertices: StaticVertex[] = [
            { position: new float3(-hx, 0, -hy), normal: n, tangent: t0, texCrd: new float2(0, 0) },
            { position: new float3(hx, 0, -hy), normal: n, tangent: t0, texCrd: new float2(1, 0) },
            { position: new float3(-hx, 0, hy), normal: n, tangent: t0, texCrd: new float2(0, 1) },
            { position: new float3(hx, 0, hy), normal: n, tangent: t0, texCrd: new float2(1, 1) },
        ];
        return { vertices, indices: new Uint32Array([2, 1, 0, 1, 2, 3]) };
    },

    createCube(size: float3 = new float3(1, 1, 1)): TriangleMeshDesc {
        const positions: number[][][] = [
            [[-0.5, -0.5, -0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, -0.5]],
            [[-0.5, 0.5, 0.5], [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5]],
            [[-0.5, 0.5, -0.5], [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5]],
            [[0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, 0.5, 0.5]],
            [[-0.5, 0.5, 0.5], [-0.5, -0.5, 0.5], [-0.5, -0.5, -0.5], [-0.5, 0.5, -0.5]],
            [[0.5, 0.5, -0.5], [0.5, -0.5, -0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5]],
        ];
        const normals: number[][] = [[0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0], [1, 0, 0]];
        const uv: number[][] = [[0, 0], [1, 0], [1, 1], [0, 1]];
        const sign = [size.x < 0 ? -1 : 1, size.y < 0 ? -1 : 1, size.z < 0 ? -1 : 1];
        const t0 = new float4(0, 0, 0, 0);
        const vertices: StaticVertex[] = [];
        const indices: number[] = [];
        for (let i = 0; i < 6; i++) {
            const idx = vertices.length;
            indices.push(idx, idx + 2, idx + 1, idx, idx + 3, idx + 2);
            for (let j = 0; j < 4; j++) {
                vertices.push({
                    position: new float3(positions[i]![j]![0]! * size.x, positions[i]![j]![1]! * size.y, positions[i]![j]![2]! * size.z),
                    normal: new float3(normals[i]![0]! * sign[0]!, normals[i]![1]! * sign[1]!, normals[i]![2]! * sign[2]!),
                    tangent: t0,
                    texCrd: new float2(uv[j]![0]!, uv[j]![1]!),
                });
            }
        }
        return { vertices, indices: new Uint32Array(indices) };
    },

    createSphere(radius = 1, segmentsU = 32, segmentsV = 32): TriangleMeshDesc {
        const t0 = new float4(0, 0, 0, 0);
        const vertices: StaticVertex[] = [];
        const indices: number[] = [];
        for (let v = 0; v <= segmentsV; v++) {
            for (let u = 0; u <= segmentsU; u++) {
                const uu = u / segmentsU;
                const vv = v / segmentsV;
                const theta = uu * 2 * Math.PI;
                const phi = vv * Math.PI;
                const dir = new float3(Math.cos(theta) * Math.sin(phi), Math.cos(phi), Math.sin(theta) * Math.sin(phi));
                vertices.push({
                    position: new float3(dir.x * radius, dir.y * radius, dir.z * radius),
                    normal: dir,
                    tangent: t0,
                    texCrd: new float2(uu, vv),
                });
            }
        }
        for (let v = 0; v < segmentsV; v++) {
            for (let u = 0; u < segmentsU; u++) {
                const i0 = v * (segmentsU + 1) + u;
                const i1 = v * (segmentsU + 1) + ((u + 1) % (segmentsU + 1));
                const i2 = (v + 1) * (segmentsU + 1) + u;
                const i3 = (v + 1) * (segmentsU + 1) + ((u + 1) % (segmentsU + 1));
                indices.push(i0, i1, i2, i2, i1, i3);
            }
        }
        return { vertices, indices: new Uint32Array(indices) };
    },

    /** Disk in the XZ plane, normal +Y (mirrors TriangleMesh::createDisk). */
    createDisk(radius = 1, segments = 32): TriangleMeshDesc {
        const n = new float3(0, 1, 0);
        const t0 = new float4(1, 0, 0, 1);
        const vertices: StaticVertex[] = [{ position: new float3(0, 0, 0), normal: n, tangent: t0, texCrd: new float2(0.5, 0.5) }];
        for (let i = 0; i <= segments; i++) {
            const a = (i / segments) * 2 * Math.PI;
            const x = Math.cos(a);
            const z = Math.sin(a);
            vertices.push({ position: new float3(x * radius, 0, z * radius), normal: n, tangent: t0, texCrd: new float2(0.5 + 0.5 * x, 0.5 + 0.5 * z) });
        }
        const indices: number[] = [];
        for (let i = 1; i <= segments; i++) indices.push(0, i + 1, i); // CW from +Y for a +Y-facing front
        return { vertices, indices: new Uint32Array(indices) };
    },
};

/** Normalizes python-side vector objects (PyProxy with x/y/z/w attrs) into
 *  owned JS vectors — proxies may be destroyed after script execution. */
export function toF3(v: { x: number; y: number; z: number } | null | undefined, fallback?: float3): float3 {
    if (!v) return fallback ?? new float3(0, 0, 0);
    return new float3(v.x, v.y, v.z);
}
export function toF4(v: { x: number; y: number; z: number; w: number } | null | undefined, fallback?: float4): float4 {
    if (!v) return fallback ?? new float4(0, 0, 0, 0);
    return new float4(v.x, v.y, v.z, v.w);
}

/**
 * Applies one deferred `getMaterial(name).<prop> = value` edit onto an imported
 * material desc; returns false for an unrecognized prop so the caller can warn.
 */
function applyMaterialEdit(mat: SceneMaterialDesc, prop: string, value: unknown): boolean {
    const header = (mat.header ??= {});
    const basic = mat.basic;
    const num = () => Number(value);
    switch (prop) {
        case "emissiveFactor": basic.emissiveFactor = num(); return true;
        case "roughness": {
            const sp = basic.specular ?? new float4(0, 1, 0, 0);
            basic.specular = new float4(sp.x, num(), sp.z, sp.w); // specular.g = roughness
            return true;
        }
        case "metallic": {
            const sp = basic.specular ?? new float4(0, 1, 0, 0);
            basic.specular = new float4(sp.x, sp.y, num(), sp.w); // specular.b = metallic
            return true;
        }
        case "indexOfRefraction": header.ior = num(); return true;
        case "specularTransmission": basic.specularTransmission = num(); return true;
        case "diffuseTransmission": basic.diffuseTransmission = num(); return true;
        case "doubleSided": header.doubleSided = Boolean(value); return true;
        case "thinSurface": header.thinSurface = Boolean(value); return true;
        case "nestedPriority": header.nestedPriority = num(); return true;
        case "volumeAbsorption": basic.volumeAbsorption = toF3(value as { x: number; y: number; z: number }); return true;
        case "volumeScattering": basic.volumeScattering = toF3(value as { x: number; y: number; z: number }); return true;
        case "baseColor": basic.baseColor = toF4(value as { x: number; y: number; z: number; w: number }); return true;
        default: return false;
    }
}

/** Camera description assembled in pyscenes (Camera() + sceneBuilder.addCamera). */
export class CameraBridge {
    private _position = new float3(0, 0, 3);
    private _target = new float3(0, 0, 0);
    private _up = new float3(0, 1, 0);
    focalLength = 21;
    focalDistance = 10000;
    apertureRadius = 0;
    /** Depth range; only the Mitsuba importer sets these (its sensor carries them). */
    nearPlane = 0.1;
    farPlane = 1000;
    shutterSpeed = 0.004;
    ISOSpeed = 100;

    set position(v: { x: number; y: number; z: number }) {
        this._position = toF3(v);
    }
    set target(v: { x: number; y: number; z: number }) {
        this._target = toF3(v);
    }
    set up(v: { x: number; y: number; z: number }) {
        this._up = toF3(v);
    }
    getPosition(): float3 {
        return this._position;
    }
    getTarget(): float3 {
        return this._target;
    }
    getUp(): float3 {
        return this._up;
    }
}

/** Material bridge mirroring the BasicMaterial python properties. Vector
 *  setters normalize python-side vectors into owned JS copies. */
export class MaterialBridge {
    private _baseColor = new float4(1, 1, 1, 1);
    /** specularParams (occlusion/roughness/metallic; BasicMaterialData default is all-zero). */
    private _specularParams = new float4(0, 0, 0, 0);
    private _transmissionColor = new float3(1, 1, 1);
    private _emissiveColor = new float3(0, 0, 0);
    emissiveFactor = 1;
    doubleSided = false;
    indexOfRefraction = 1.5;
    specularTransmission = 0;
    diffuseTransmission = 0;
    thinSurface = false;
    nestedPriority = 0;
    displacementScale = 0;
    displacementOffset = 0;
    private _volumeAbsorption = new float3(0, 0, 0);
    private _volumeScattering = new float3(0, 0, 0);

    set volumeAbsorption(v: { x: number; y: number; z: number }) {
        this._volumeAbsorption = toF3(v);
    }
    set volumeScattering(v: { x: number; y: number; z: number }) {
        this._volumeScattering = toF3(v);
    }

    constructor(
        public readonly materialType: MaterialType,
        public readonly name: string,
        shadingModel: ShadingModel = ShadingModel.MetalRough,
    ) {
        this._shadingModel = shadingModel;
    }

    /** StandardMaterial's shading model (fixed at construction, like native). */
    private _shadingModel: ShadingModel;
    get shadingModel(): ShadingModel {
        return this._shadingModel;
    }

    /** Native's metal-rough-only setters refuse spec-gloss StandardMaterials. */
    private rejectsMetalRoughParam(what: string): boolean {
        if (this.materialType !== MaterialType.Standard || this._shadingModel === ShadingModel.MetalRough) return false;
        Logger.warning(`Ignoring set${what}(). Material '${this.name}' does not use the metallic/roughness shading model.`);
        return true;
    }

    // Deferred texture loads (material.loadTexture(slot, path)); resolved in resolve().
    private _textures: { slot: string; path: string }[] = [];
    /** Measured BRDF file (MERL `.binary` / RGL `.bsdf`), resolved with the textures. */
    private _measured: { kind: "merl" | "rgl"; path: string } | null = null;
    private _lightProfileEnabled = false;
    /** MERLMix takes a *list* of BRDF paths plus a per-texel index map. */
    private _merlMixPaths: string[] = [];
    private _merl: import("./Material/MERLFile.js").MERLBRDF | null = null;
    private _rgl: import("./Material/RGLFile.js").RGLMeasurement | null = null;
    private _merlMix: import("./Material/MERLFile.js").MERLMixData | null = null;
    private _bitmaps: { slot: string; bitmap: ImageBitmap; srgb: boolean }[] = [];
    private _indexMap: import("./Material/MERLFile.js").MERLIndexMap | null = null;
    private _texHandles: { texBaseColor?: number; texSpecular?: number; texEmissive?: number; texNormalMap?: number; texDisplacement?: number; texTransmission?: number } = {};

    loadTexture(slot: string, path: string): void {
        this._textures.push({ slot: String(slot), path: String(path) });
    }

    /** Binds an already-decoded image (procedural content: Mitsuba's checkerboard). */
    loadTextureBitmap(slot: string, bitmap: ImageBitmap, srgb: boolean): void {
        this._bitmaps.push({ slot: String(slot), bitmap, srgb });
    }

    /** Fetches + decodes this material's deferred textures into the TextureManager (MaterialTextureLoader: sRGB for colour slots unless AssumeLinearSpaceTextures). */
    async resolveTextures(baseUrl: string, tm: TextureManager, resolver = AssetResolver.getDefaultResolver(), assumeLinearSpaceTextures = false): Promise<void> {
        for (const t of this._textures) {
            const url = await resolveAssetUrl(t.path, baseUrl, AssetCategory.Any, resolver);
            // MERLMix's index map never reaches the texture array: BRDF indices
            // must be point-sampled and the packed array shares a linear sampler,
            // so the bytes go into the material buffer instead (docs §9).
            if (t.slot === "Index" && this.materialType === MaterialType.MERLMix) {
                this._indexMap = await loadIndexMap(url, t.path);
                continue;
            }
            // Mirrors MaterialTextureLoader::loadTexture: the material's own slot
            // table decides whether the slot exists and whether it is sRGB.
            const slotSrgb = getTextureSlotSrgb(this.materialType, this._shadingModel, t.slot);
            if (slotSrgb === undefined) {
                Logger.warning(`MaterialTextureLoader::loadTexture() - Material '${this.name}' does not have texture slot '${t.slot}'. Ignoring call.`);
                continue;
            }
            try {
                const res = await fetch(url);
                if (!res.ok) continue;
                const srgb = slotSrgb && !assumeLinearSpaceTextures;
                const blob = await res.blob();
                const bytes = new Uint8Array(await blob.arrayBuffer());
                // Formats the browser cannot decode go through the CPU decoders
                // (native gets these from FreeImage).
                const bitmap = t.path.toLowerCase().endsWith(".tga")
                    ? await decodeTgaToBitmap(bytes)
                    : await createImageBitmap(blob, { colorSpaceConversion: "none" });
                this.assignTextureHandle(t.slot, packTextureHandle(TextureHandleMode.Texture, tm.addTexture({ bitmap, srgb, bytes })));
            } catch {
                /* undecodable format (e.g. DDS) — material falls back to base color */
            }
        }
        for (const b of this._bitmaps) {
            const handle = packTextureHandle(TextureHandleMode.Texture, tm.addTexture({ bitmap: b.bitmap, srgb: b.srgb }));
            this.assignTextureHandle(b.slot, handle);
        }
    }

    /** Routes a packed texture handle to the slot's field. */
    private assignTextureHandle(slot: string, handle: number): void {
        if (slot === "BaseColor") this._texHandles.texBaseColor = handle;
        else if (slot === "Specular") this._texHandles.texSpecular = handle;
        else if (slot === "Normal") this._texHandles.texNormalMap = handle;
        else if (slot === "Emissive") this._texHandles.texEmissive = handle;
        else if (slot === "Displacement") this._texHandles.texDisplacement = handle;
        else if (slot === "Transmission") this._texHandles.texTransmission = handle;
    }

    set baseColor(v: { x: number; y: number; z: number; w: number }) {
        this._baseColor = toF4(v);
    }
    set specularParams(v: { x: number; y: number; z: number; w: number }) {
        this._specularParams = toF4(v);
    }
    set transmissionColor(v: { x: number; y: number; z: number }) {
        this._transmissionColor = toF3(v);
    }
    set emissiveColor(v: { x: number; y: number; z: number }) {
        this._emissiveColor = toF3(v);
    }

    /** ClothMaterial/BasicMaterial::setRoughness -> specular.g. */
    set roughness(r: number | { x: number; y: number }) {
        if (this.rejectsMetalRoughParam("Roughness")) return;
        if (typeof r === "number") {
            this._specularParams = new float4(this._specularParams.x, r, this._specularParams.z, this._specularParams.w);
        } else {
            // PBRTConductorMaterial::setRoughness(float2) -> specular.rg.
            this._specularParams = new float4(r.x, r.y, this._specularParams.z, this._specularParams.w);
        }
    }

    /** StandardMaterial::setMetallic -> specular.b. */
    set metallic(m: number) {
        if (this.rejectsMetalRoughParam("Metallic")) return;
        this._specularParams = new float4(this._specularParams.x, this._specularParams.y, m, this._specularParams.w);
    }

    /** Mirrors StandardMaterial::setLightProfileEnabled. */
    set lightProfileEnabled(enabled: boolean) {
        this._lightProfileEnabled = !!enabled;
    }

    /** Mirrors RGLMaterial::loadBRDF / the MERLMaterial(path) constructor. */
    load(path: unknown): void {
        if (this.materialType === MaterialType.MERLMix) {
            this._merlMixPaths.push(String(path));
            return;
        }
        this._measured = { kind: this.materialType === MaterialType.RGL ? "rgl" : "merl", path: String(path) };
    }

    /** Fetches the measured BRDF, alongside resolveTextures (docs §9: async asset IO). */
    async resolveMeasured(baseUrl: string, resolver = AssetResolver.getDefaultResolver()): Promise<void> {
        if (this._merlMixPaths.length > 0) {
            const { loadMERLBinary } = await import("./Material/MERLFile.js");
            const brdfs = [];
            for (const path of this._merlMixPaths) {
                brdfs.push(await loadMERLBinary(await resolveAssetUrl(path, baseUrl, AssetCategory.Any, resolver)));
            }
            // Mirrors MERLMixMaterial's constructor: without an index map every
            // texel selects BRDF 0.
            const indexMap = this._indexMap ?? { width: 1, height: 1, indices: new Uint8Array(1) };
            this._merlMix = { brdfs, indexMap, texNormalMap: this._texHandles.texNormalMap };
            return;
        }
        if (!this._measured) return;
        const url = await resolveAssetUrl(this._measured.path, baseUrl, AssetCategory.Any, resolver);
        if (this._measured.kind === "rgl") {
            const { loadRGLFile } = await import("./Material/RGLFile.js");
            this._rgl = await loadRGLFile(url);
        } else {
            const { loadMERLBinary } = await import("./Material/MERLFile.js");
            this._merl = await loadMERLBinary(url);
        }
    }

    toDesc(): SceneMaterialDesc {
        const emissive = this._emissiveColor.x !== 0 || this._emissiveColor.y !== 0 || this._emissiveColor.z !== 0;
        if (this._merlMix) return { name: this.name, header: { materialType: MaterialType.MERLMix }, basic: {}, merlMix: this._merlMix };
        if (this._merl) return { name: this.name, header: { materialType: MaterialType.MERL }, basic: {}, merl: this._merl };
        if (this._rgl) return { name: this.name, header: { materialType: MaterialType.RGL }, basic: {}, rgl: this._rgl };
        return {
            name: this.name,
            header: {
                materialType: this.materialType,
                lightProfileEnabled: this._lightProfileEnabled,
                doubleSided: this.doubleSided,
                emissive,
                ior: this.indexOfRefraction,
                thinSurface: this.thinSurface,
                nestedPriority: this.nestedPriority,
            },
            basic: {
                baseColor: this._baseColor,
                specular: this._specularParams,
                transmission: this._transmissionColor,
                emissive: this._emissiveColor,
                emissiveFactor: this.emissiveFactor,
                specularTransmission: this.specularTransmission,
                diffuseTransmission: this.diffuseTransmission,
                volumeAbsorption: this._volumeAbsorption,
                volumeScattering: this._volumeScattering,
                displacementScale: this.displacementScale,
                displacementOffset: this.displacementOffset,
                ...(this.materialType === MaterialType.Standard ? { shadingModel: this._shadingModel } : {}),
                ...this._texHandles,
            },
        };
    }
}

export class LightBridge {
    private _position = new float3(0, 0, 0);
    private _intensity = new float3(1, 1, 1);
    private _direction = new float3(0, -1, 0);
    /** DistantLight: half-angle (radians); default = sun (DistantLight ctor). */
    angle = 0.5 * 0.53 * (Math.PI / 180);
    /** PointLight spot cone: cutoff half-angle (PI = omnidirectional) + penumbra. */
    openingAngle = Math.PI;
    penumbraAngle = 0;
    /** Area lights (Rect/Disc/Sphere): local->world placement (scale/rotate/translate). */
    private _scaling: float3 | number = 1;
    private _rotationEuler: { x: number; y: number; z: number } | null = null;
    constructor(
        public readonly lightType: LightType,
        public readonly name: string,
    ) {}

    set position(v: { x: number; y: number; z: number }) {
        this._position = toF3(v);
    }
    set intensity(v: { x: number; y: number; z: number }) {
        this._intensity = toF3(v);
    }
    set direction(v: { x: number; y: number; z: number }) {
        this._direction = toF3(v);
    }
    set scaling(v: { x: number; y: number; z: number } | number) {
        this._scaling = typeof v === "number" ? v : toF3(v);
    }
    set rotation(v: { x: number; y: number; z: number }) {
        this._rotationEuler = toF3(v);
    }
    getPosition(): float3 {
        return this._position;
    }
    getIntensity(): float3 {
        return this._intensity;
    }
    getDirection(): float3 {
        return this._direction;
    }
    /** Area-light transform matrix (transMat = T * R * S; native folds scaling in). */
    getTransMat(): float4x4 {
        return makeTransform(this._position, this._rotationEuler, null, this._scaling);
    }
}

/** Transform bridge (composition order Translate * Rotate * Scale, as native). */
type VecLike = { x: number; y: number; z: number };
export function makeTransform(
    translationIn: VecLike | null,
    rotationEulerIn: VecLike | null,
    rotationEulerDegIn: VecLike | null,
    scalingIn: VecLike | number | null,
): float4x4 {
    // Scalar scaling broadcasts (native float3 constructor from scalar);
    // python-side vectors normalize to owned JS copies.
    const scaling = typeof scalingIn === "number" ? new float3(scalingIn, scalingIn, scalingIn) : scalingIn ? toF3(scalingIn) : null;
    const translation = translationIn ? toF3(translationIn) : null;
    const rotationEuler = rotationEulerIn ? toF3(rotationEulerIn) : null;
    const rotationEulerDeg = rotationEulerDegIn ? toF3(rotationEulerDegIn) : null;
    const rot = rotationEuler ?? (rotationEulerDeg ? new float3((rotationEulerDeg.x * Math.PI) / 180, (rotationEulerDeg.y * Math.PI) / 180, (rotationEulerDeg.z * Math.PI) / 180) : null);
    let m = float4x4.identity();
    if (scaling) m = mulMat(matrixFromScaling(scaling), m);
    if (rot) {
        // R = Rz * Ry * Rx — numerically verified against math::quatFromEulerAngles
        // (Falcor's euler quat expands to the ZYX matrix product).
        const [cx, sx, cy, sy, cz, sz] = [Math.cos(rot.x), Math.sin(rot.x), Math.cos(rot.y), Math.sin(rot.y), Math.cos(rot.z), Math.sin(rot.z)];
        const r = float4x4.identity();
        r.set(0, 0, cy * cz);
        r.set(0, 1, sx * sy * cz - cx * sz);
        r.set(0, 2, cx * sy * cz + sx * sz);
        r.set(1, 0, cy * sz);
        r.set(1, 1, sx * sy * sz + cx * cz);
        r.set(1, 2, cx * sy * sz - sx * cz);
        r.set(2, 0, -sy);
        r.set(2, 1, sx * cy);
        r.set(2, 2, cx * cy);
        m = mulMat(r, m);
    }
    if (translation) m = mulMat(matrixFromTranslation(translation), m);
    return m;
}

export type { SDFGridType, SDFGridRecipe };

/** Mirrors SceneBuilder::Flags (values identical to native; see docs §8.4 for which ones change web behaviour). */
export enum SceneBuilderFlags {
    None = 0x0,
    DontMergeMaterials = 0x1,
    UseOriginalTangentSpace = 0x2,
    AssumeLinearSpaceTextures = 0x4,
    DontMergeMeshes = 0x8,
    UseSpecGlossMaterials = 0x10,
    UseMetalRoughMaterials = 0x20,
    NonIndexedVertices = 0x40,
    Force32BitIndices = 0x80,
    RTDontMergeStatic = 0x100,
    RTDontMergeDynamic = 0x200,
    RTDontMergeInstanced = 0x400,
    FlattenStaticMeshInstances = 0x800,
    DontOptimizeGraph = 0x1000,
    DontOptimizeMaterials = 0x2000,
    DontUseDisplacement = 0x4000,
    UseCompressedHitInfo = 0x8000,
    TessellateCurvesIntoPolyTubes = 0x10000,
    UseCache = 0x10000000,
    RebuildCache = 0x20000000,
    Default = None,
}

/** Python enum surface (`SceneBuilderFlags.NonIndexedVertices | ...`). */
export const kSceneBuilderFlagsPython: Record<string, number> = Object.fromEntries(
    Object.entries(SceneBuilderFlags).filter(([, v]) => typeof v === "number") as [string, number][],
);

/** Importer options derived from the build flags. */
export interface ImportOptions {
    /** Flags::AssumeLinearSpaceTextures: colour textures are not sRGB-decoded. */
    assumeLinearSpaceTextures?: boolean;
}

/** Recorded SDF grid state (mirrors SDFGrid python bindings; ND + SBS types). */
export class SDFGridBridge {
    ops: SDFGridRecipe["ops"] = [];
    /** Pending loadValuesFromFile/loadPrimitivesFromFile calls, fetched in resolve(). */
    pendingFiles: { path: string; primitives?: boolean; gridWidth?: number }[] = [];
    constructor(
        readonly type: SDFGridType,
        readonly narrowBandThickness: number,
        readonly brickWidth: number,
    ) {}
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.ops.push({ kind: "cheese", gridWidth: Number(gridWidth), seed: Number(seed) });
    }

    /** Mirrors SDFGrid::loadValuesFromFile (the `.sdfg` corner-value format). */
    loadValuesFromFile(path: unknown): boolean {
        this.pendingFiles.push({ path: String(path) });
        return true;
    }

    /** Mirrors SDFGrid::loadPrimitivesFromFile (the `.sdf` primitive-list format). */
    loadPrimitivesFromFile(path: unknown, gridWidth: unknown): number {
        this.pendingFiles.push({ path: String(path), primitives: true, gridWidth: Number(gridWidth) });
        // Native returns the primitive count; it is only known after the fetch.
        return 0;
    }
    toRecipe(): SDFGridRecipe {
        return { type: this.type, narrowBandThickness: this.narrowBandThickness, brickWidth: this.brickWidth, ops: [...this.ops] };
    }
}

/**
 * Mirrors SceneBuilder::processMesh under Flags::NonIndexedVertices: vertex
 * data is expanded so vertex i is the one index i referenced. Native then drops
 * the index buffer; the port keeps an identity one, which addresses the same
 * vertices in the same order. Per-vertex skinning and morph data expand too.
 */
export function deindexMesh(mesh: SceneMeshDesc): SceneMeshDesc {
    const n = mesh.indices.length;
    const vertices = Array.from(mesh.indices, (i) => ({ ...mesh.vertices[i]! }));
    const expand = (data: Float32Array | Uint32Array, lanes: number) => {
        const out = new (data.constructor as Float32ArrayConstructor | Uint32ArrayConstructor)(n * lanes);
        for (let v = 0; v < n; v++) out.set(data.subarray(mesh.indices[v]! * lanes, mesh.indices[v]! * lanes + lanes), v * lanes);
        return out;
    };
    return {
        ...mesh,
        vertices,
        indices: Uint32Array.from({ length: n }, (_v, i) => i),
        skin: mesh.skin ? { ...mesh.skin, boneIDs: expand(mesh.skin.boneIDs, 4) as Uint32Array, weights: expand(mesh.skin.weights, 4) as Float32Array } : undefined,
        morph: mesh.morph
            ? {
                  ...mesh.morph,
                  targets: mesh.morph.targets.map((t) => ({
                      position: expand(t.position, 3) as Float32Array,
                      ...(t.normal ? { normal: expand(t.normal, 3) as Float32Array } : {}),
                  })),
              }
            : undefined,
    };
}

/** Decodes a TGA and hands back an ImageBitmap, as the browser decoders do. */
async function decodeTgaToBitmap(bytes: Uint8Array): Promise<ImageBitmap> {
    const { decodeTGA } = await import("../Utils/Image/TGADecoder.js");
    const image = decodeTGA(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
    const pixels = new Uint8ClampedArray(image.rgba.length);
    pixels.set(image.rgba);
    const data = new ImageData(pixels, image.width, image.height);
    return createImageBitmap(data, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
}

/**
 * Reads a MERLMix index map as raw 8-bit indices (upstream samples the red
 * channel of an 8-bit unorm texture and scales it by 255).
 */
async function loadIndexMap(url: string, path: string): Promise<import("./Material/MERLFile.js").MERLIndexMap> {
    const bytes = new Uint8Array(await (await fetch(url)).arrayBuffer());
    let width: number;
    let height: number;
    let rgba: Uint8Array | Uint8ClampedArray;
    if (path.toLowerCase().endsWith(".tga")) {
        const { decodeTGA } = await import("../Utils/Image/TGADecoder.js");
        const image = decodeTGA(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer);
        ({ width, height } = image);
        rgba = image.rgba;
    } else {
        const bitmap = await createImageBitmap(new Blob([bytes as BlobPart]), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
        width = bitmap.width;
        height = bitmap.height;
        const canvas = new OffscreenCanvas(width, height);
        const g = canvas.getContext("2d", { willReadFrequently: true })!;
        g.drawImage(bitmap, 0, 0);
        rgba = g.getImageData(0, 0, width, height).data;
    }
    const indices = new Uint8Array(width * height);
    for (let i = 0; i < indices.length; i++) indices[i] = rgba[i * 4]!;
    return { width, height, indices };
}

/** Recorded GridVolume state (eager copies; PyProxies die at script exit). */
export class GridVolumeBridge {
    name: string;
    densityScale = 1;
    emissionScale = 1;
    albedo: { x: number; y: number; z: number } = { x: 1, y: 1, z: 1 };
    anisotropy = 0;
    emissionTemperature = 0;
    grids: { slot: string; path: string; gridname: string }[] = [];
    /** Pending loadGridSequence calls: several files feeding one slot. */
    gridSequences: { slot: string; paths: string[]; gridname: string }[] = [];
    frameRate = 30;
    startFrame = 0;
    playbackEnabled = true;
    proceduralGrids: { slot: string; parsed: ParsedFloatGrid }[] = [];

    constructor(name = "") {
        this.name = String(name);
    }

    /** Mirrors GridVolume::loadGridSequence(slot, paths, gridname). */
    loadGridSequence(slot: unknown, paths: unknown, gridname: unknown): number {
        // A python list arrives as a proxy, not a JS array; both are iterable.
        const iterable = paths as Iterable<unknown> | null;
        const list = (typeof paths === "string" || !iterable || typeof iterable[Symbol.iterator] !== "function" ? [paths] : [...iterable]).map((p) => String(p));
        this.gridSequences.push({ slot: String(slot), paths: list, gridname: String(gridname) });
        return list.length;
    }

    loadGrid(slot: unknown, path: unknown, gridname: unknown): boolean {
        this.grids.push({ slot: String(slot), path: String(path), gridname: String(gridname) });
        return true;
    }

    /** volume.densityGrid = Grid.createSphere/createBox(...) — a procedural grid. */
    set densityGrid(g: { _proceduralGrid?: ParsedFloatGrid } | null) {
        if (g?._proceduralGrid) this.proceduralGrids.push({ slot: "Density", parsed: g._proceduralGrid });
    }
}

interface EnvMapRef {
    /** Empty when the env map is a constant colour (Mitsuba's `constant` emitter). */
    path: string;
    intensity: number;
    rotation?: { x: number; y: number; z: number };
    /** A uniform radiance, uploaded as the 1x1 texture native builds for it. */
    constantColor?: [number, number, number];
    /** The file is an equal-area octahedral map (pbrt-v4 `infinite` lights). */
    equalAreaOctahedral?: boolean;
}

type Command =
    | { kind: "import"; path: string }
    | { kind: "mesh"; mesh: TriangleMeshDesc; material: MaterialBridge; nodeTransform: float4x4 };

export class SceneBuilderBridge {
    /** Copy of the default resolver at construction (native SceneBuilder::mAssetResolver). */
    readonly assetResolver = AssetResolver.getDefaultResolver().clone();

    constructor(readonly flags: SceneBuilderFlags = SceneBuilderFlags.Default) {}
    getFlags(): SceneBuilderFlags {
        return this.flags;
    }
    private hasFlag(flag: SceneBuilderFlags): boolean {
        return (this.flags & flag) !== 0;
    }
    private get importOptions(): ImportOptions {
        return { assumeLinearSpaceTextures: this.hasFlag(SceneBuilderFlags.AssumeLinearSpaceTextures) };
    }
    getAssetResolver(): AssetResolver {
        return this.assetResolver;
    }
    private commands: Command[] = [];
    private meshMaterials: MaterialBridge[] = []; // by meshID
    private meshGeometry: TriangleMeshDesc[] = [];
    private meshInstanced = new Map<number, float4x4[]>();
    private nodes: float4x4[] = [];
    private lights: LightBridge[] = [];
    /** Node driving an imported camera (glTF), for camera animation; undefined if none. */
    private importedCameraNodeID: number | undefined;
    /** Bind-pose of an imported glTF camera (used when the pyscene sets no camera). */
    private importedCameraPose: import("./Importer/GltfImporter.js").GltfCameraPose | undefined;
    private gridVolumesList: GridVolumeBridge[] = [];
    private _envMap: EnvMapRef | null = null;
    camera: CameraBridge | null = null;

    /** Eagerly copies the descriptor: python-side values (PyProxies) may not
     *  outlive the script, and props are set before assignment in pyscenes. */
    set envMap(v: EnvMapRef | null) {
        this._envMap = v
            ? {
                  path: String(v.path),
                  intensity: Number(v.intensity),
                  rotation: v.rotation ? { x: Number(v.rotation.x), y: Number(v.rotation.y), z: Number(v.rotation.z) } : undefined,
                  constantColor: v.constantColor ? [Number(v.constantColor[0]), Number(v.constantColor[1]), Number(v.constantColor[2])] : undefined,
                  equalAreaOctahedral: v.equalAreaOctahedral ? true : undefined,
              }
            : null;
    }
    get envMap(): EnvMapRef | null {
        return this._envMap;
    }
    cameraSpeed = 1;
    private _cameras: CameraBridge[] = [];

    addCamera(camera: CameraBridge): void {
        const c = unwrapGuard(camera);
        this._cameras.push(c);
        this.camera = c; // default selection = most recent; overridden by selectedCamera
    }
    get cameras(): CameraBridge[] {
        return this._cameras;
    }
    set selectedCamera(camera: CameraBridge) {
        this.camera = unwrapGuard(camera);
    }

    /** Deferred animation handles: imports resolve later, so pyscene behavior
     *  writes are recorded here and applied to the imported clips in resolve()
     *  (index = native Animation creation order = assimp node-anim order). */
    private animationHandles: { preInfinityBehavior: unknown; postInfinityBehavior: unknown }[] = [];
    get animations(): { preInfinityBehavior: unknown; postInfinityBehavior: unknown }[] {
        if (this.animationHandles.length === 0) {
            this.animationHandles = Array.from({ length: 64 }, () => ({ preInfinityBehavior: null, postInfinityBehavior: null }));
        }
        return this.animationHandles;
    }

    importScene(path: string): void {
        this.commands.push({ kind: "import", path });
    }

    addTriangleMesh(mesh: TriangleMeshDesc, material: MaterialBridge): number {
        this.meshGeometry.push(mesh);
        this.meshMaterials.push(unwrapGuard(material));
        return this.meshGeometry.length - 1;
    }

    addNode(_name: string, transform: float4x4, parentID?: number): number {
        // Nodes store WORLD matrices (static scenes): compose under the parent
        // (native SceneBuilder::addNode's third argument was silently dropped
        // before — parented pyscene nodes lost the parent transform).
        const parent = parentID !== undefined && parentID >= 0 ? this.nodes[parentID] : undefined;
        this.nodes.push(parent ? mulMat(parent, transform) : transform);
        return this.nodes.length - 1;
    }

    /** Curve geometry added by importers (linear swept spheres), with its material. */
    private builderCurves: { positionsRadii: Float32Array; indices: Uint32Array; material: MaterialBridge; nodeID: number }[] = [];

    /**
     * Mirrors SceneBuilder::addCurve + addCurveInstance: tessellated curve
     * geometry (xyz + radius per vertex, segment-start indices) placed by a node.
     */
    addCurveInstance(nodeID: number, curve: { positionsRadii: Float32Array; indices: Uint32Array }, material: MaterialBridge): void {
        if (!this.nodes[nodeID]) throw new RuntimeError(`addCurveInstance: unknown node ${nodeID}`);
        this.builderCurves.push({ ...curve, material, nodeID });
    }

    addMeshInstance(nodeID: number, meshID: number): void {
        const transform = this.nodes[nodeID];
        if (!transform) throw new RuntimeError(`addMeshInstance: unknown node ${nodeID}`);
        // One mesh may be instanced under many nodes (e.g. nested_dielectrics
        // instances one cube 30x) — accumulate, don't overwrite.
        const list = this.meshInstanced.get(meshID);
        if (list) list.push(transform);
        else this.meshInstanced.set(meshID, [transform]);
    }

    /** Recorded SceneBuilder::addCustomPrimitive calls (user ID + AABB). */
    private customPrimitives: { userID: number; aabb: { min: [number, number, number]; max: [number, number, number] } }[] = [];

    /**
     * Mirrors SceneBuilder::addCustomPrimitive. Like native, the primitive is an
     * AABB with a user ID for passes that supply their own intersection code; the
     * shipped passes have none, so it does not render (it used to be drawn as a
     * box mesh here).
     */
    addCustomPrimitive(userID: number, aabb: { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } }): void {
        const { min, max } = aabb;
        this.customPrimitives.push({
            userID: Number(userID),
            aabb: { min: [Number(min.x), Number(min.y), Number(min.z)], max: [Number(max.x), Number(max.y), Number(max.z)] },
        });
    }

    addLight(light: LightBridge): void {
        this.lights.push(unwrapGuard(light));
    }

    /** Deferred material edits from getMaterial() (imports resolve later). */
    materialEdits: { name: string; prop: string; value: unknown }[] = [];

    /**
     * Multiplier applied to every imported material's emissiveFactor in resolve() —
     * the web analog of the pyscene idiom that boosts all emissives (e.g. `*= 1000`).
     */
    globalEmissiveScale = 1;
    boostAllEmissive(scale: number): void {
        this.globalEmissiveScale *= Number(scale);
    }

    /**
     * Yields one stand-in backing emissiveFactor with globalEmissiveScale, for the
     * `for m in sceneBuilder.materials: m.emissiveFactor *= N` idiom; other writes warn.
     */
    get materials(): unknown[] {
        const bridge = this;
        const proxy = new Proxy(
            {},
            {
                get(_t, prop) {
                    if (typeof prop === "symbol") return undefined; // let pyodide probe iterator/thenable
                    if (prop === "emissiveFactor") return bridge.globalEmissiveScale;
                    return undefined; // other props unknown before import resolves
                },
                set(_t, prop, value) {
                    if (prop === "emissiveFactor") {
                        bridge.globalEmissiveScale = Number(value);
                        return true;
                    }
                    Logger.warning(`sceneBuilder.materials[*].${String(prop)}: per-material writes over the whole list aren't supported pre-import; ignored`);
                    return true;
                },
            },
        );
        return [proxy];
    }

    getMaterial(name: string): unknown {
        const edits = this.materialEdits;
        const matName = String(name);
        // Recorder handle: property writes apply after the import resolves.
        return new Proxy(
            {},
            {
                set(_t, prop, value) {
                    // Keep the raw value: bool (doubleSided) and float3 (volumeAbsorption)
                    // props must survive to resolve(); Number() would mangle them.
                    edits.push({ name: matName, prop: String(prop), value });
                    return true;
                },
                get(_t, prop) {
                    throw new RuntimeError(`SceneBuilder.getMaterial('${matName}').${String(prop)}: reads are unsupported on the web bridge (deferred import)`);
                },
            },
        );
    }

    private sdfGridsList: { grid: SDFGridBridge; material: MaterialBridge }[] = [];
    private sdfInstances: { nodeID: number; sdfGridID: number }[] = [];
    /** Pending SceneBuilder::loadLightProfile call, resolved (and baked) in resolve(). */
    private lightProfileRequest: { path: string; normalize: boolean } | null = null;

    /**
     * Mirrors SceneBuilder::loadLightProfile: the IES profile is shared by every
     * material with `lightProfileEnabled`. Resolved and baked in resolve().
     */
    loadLightProfile(filename: unknown, normalize: unknown = true): void {
        this.lightProfileRequest = { path: String(filename), normalize: normalize !== false };
    }


    addSDFGrid(grid: SDFGridBridge, material: MaterialBridge): number {
        const g = unwrapGuard(grid) as SDFGridBridge;
        const copy = new SDFGridBridge(g.type, Number(g.narrowBandThickness), Number(g.brickWidth));
        copy.ops = g.ops.map((o) => ({ ...o }));
        copy.pendingFiles = g.pendingFiles.map((f) => ({ ...f }));
        this.sdfGridsList.push({ grid: copy, material: unwrapGuard(material) });
        return this.sdfGridsList.length - 1;
    }

    addSDFGridInstance(nodeID: number, sdfGridID: number): void {
        if (!this.nodes[nodeID]) throw new RuntimeError(`addSDFGridInstance: unknown node ${nodeID}`);
        this.sdfInstances.push({ nodeID: Number(nodeID), sdfGridID: Number(sdfGridID) });
    }

    /** Legacy alias used by some pyscenes (volume_transmittance_test). */
    addVolume(volume: GridVolumeBridge): void {
        this.addGridVolume(volume);
    }
    addGridVolume(volume: GridVolumeBridge): void {
        const v = unwrapGuard(volume);
        // Eager-copy scalar props (albedo may be a python float3 proxy).
        const copy = new GridVolumeBridge(v.name);
        copy.densityScale = Number(v.densityScale);
        copy.emissionScale = Number(v.emissionScale);
        copy.albedo = { x: Number(v.albedo.x), y: Number(v.albedo.y), z: Number(v.albedo.z) };
        copy.anisotropy = Number(v.anisotropy);
        copy.emissionTemperature = Number(v.emissionTemperature);
        copy.grids = v.grids.map((g) => ({ slot: String(g.slot), path: String(g.path), gridname: String(g.gridname) }));
        copy.gridSequences = v.gridSequences.map((g) => ({ slot: String(g.slot), paths: g.paths.map((path) => String(path)), gridname: String(g.gridname) }));
        copy.frameRate = Number(v.frameRate);
        copy.startFrame = Number(v.startFrame);
        copy.playbackEnabled = v.playbackEnabled !== false;
        copy.proceduralGrids = v.proceduralGrids.slice();
        this.gridVolumesList.push(copy);
    }

    /** Fetches referenced assets and constructs the Scene. */
    /** Scene ctor args snapshot from the last resolve() (SceneCache capture). */
    lastSceneArgs: {
        meshes: SceneMeshDesc[];
        materials: SceneMaterialDesc[];
        lights: AnalyticLight[];
        nodes: SceneNode[];
        cameraNodeID?: number;
        textureManager: TextureManager;
        curves: import("./Scene.js").SceneCurveDesc[];
        animations: AnimationChannel[];
        weightTracks: WeightTrack[];
        /** SDF grids as rebuildable recipes + their instances (SceneCache v4). */
        sdfGrids: { recipes: SDFGridRecipe[]; instances: { gridIndex: number; materialID: number; transform?: float4x4 }[] };
        cacheable: boolean;
    } | null = null;

    async resolve(device: Device, baseUrl: string): Promise<Scene> {
        const textureManager = new TextureManager();
        const meshes: SceneMeshDesc[] = [];
        const materials: SceneMaterialDesc[] = [];
        const nodes: SceneNode[] = []; // retained scene-graph nodes (for animation)
        const animations: AnimationChannel[] = [];
        const weightTracks: WeightTrack[] = []; // morph-weight tracks from glTF imports
        const importedLights: AnalyticLight[] = []; // lights from imported assets (FBX)

        const importedMaterialNames: string[] = [];
        const curves: import("./Scene.js").SceneCurveDesc[] = [];
        let clipOffset = 0; // clip ordinals accumulate across imports (native Animation list order)
        for (const cmd of this.commands) {
            if (cmd.kind === "import") {
                const url = await resolveAssetUrl(cmd.path, baseUrl, AssetCategory.Scene, this.assetResolver);
                const res = await fetch(url);
                if (!res.ok) throw new RuntimeError(`SceneBuilder: Can't find scene file '${cmd.path}' (tried '${url}', ${res.status})`);
                const bytes = new Uint8Array(await res.arrayBuffer());
                const materialOffset = materials.length;
                if (/\.usd[acz]?$/.test(cmd.path.toLowerCase())) {
                    const dir = url.slice(0, url.lastIndexOf("/"));
                    // Curve prims import as real curves below; tinyusdz also
                    // tessellates them into meshes — exclude those duplicates.
                    const curvePrims = cmd.path.toLowerCase().endsWith(".usda")
                        ? extractBasisCurvesFromUsda(new TextDecoder().decode(bytes))
                        : [];
                    const parsed = await UsdImporter.parseToDescs(bytes, textureManager, dir, new Set(curvePrims.map((c) => c.name)), this.importOptions);
                    materials.push(...parsed.materials);
                    importedMaterialNames.push(...parsed.materialNames);
                    for (const m of parsed.meshes) meshes.push({ ...m, materialID: m.materialID + materialOffset });
                    // BasisCurves from USDA text (tinyusdz's RenderScene has no curve API).
                    {
                        const strands = curvePrims;
                        if (strands.length > 0) {
                            const materialID = materials.length;
                            // Native default curve material (ImporterContext): Hair,
                            // baseColor (0.8,0.4,0.05), specular (longRough, azimRough,
                            // scaleAngleDeg, 0), IOR 1.55.
                            materials.push({
                                name: "default-curve-0",
                                header: { materialType: MaterialType.Hair, ior: 1.55 },
                                basic: { baseColor: new float4(0.8, 0.4, 0.05, 1), specular: new float4(0.125, 0.3, 1, 0) },
                            });
                            importedMaterialNames.push("default-curve-0");
                            // Dynamic import: Scripting imports this module.
                            const settings = (await import("../Utils/Scripting/Scripting.js")).getGlobalSettings();
                            for (const strand of strands) {
                                // Per-prim Settings attributes, keyed by prim path as natively
                                // (ImporterContext's "curves:*" attributes).
                                const subdiv = Number(settings.getAttribute(strand.path, "curves:subdivPerSegment", 1));
                                const keepStrands = Number(settings.getAttribute(strand.path, "curves:keepOneEveryXStrands", 1));
                                const keepVertices = Number(settings.getAttribute(strand.path, "curves:keepOneEveryXVerticesPerStrand", 1));
                                // Fewer strands render wider, to keep the perceived density.
                                const widthScale = Math.sqrt(keepStrands);
                                let polytube = this.hasFlag(SceneBuilderFlags.TessellateCurvesIntoPolyTubes);
                                const mode = String(settings.getAttribute<string>(strand.path, "curves:mode", ""));
                                if (mode === "lss") polytube = false;
                                else if (mode === "polytube") polytube = true;

                                const strandCount = strand.curveVertexCounts.length;
                                if (polytube) {
                                    // CurveTessellationMode::PolyTube: the curve becomes a
                                    // triangle mesh (4-gon cross sections, as natively).
                                    const tube = convertToPolytube(strandCount, strand.curveVertexCounts, strand.points, strand.widths, null, subdiv, keepStrands, keepVertices, widthScale, 4);
                                    const vertices: StaticVertex[] = [];
                                    for (let v = 0; v < tube.radii.length; v++) {
                                        vertices.push({
                                            position: new float3(tube.vertices[v * 3]!, tube.vertices[v * 3 + 1]!, tube.vertices[v * 3 + 2]!),
                                            normal: new float3(tube.normals[v * 3]!, tube.normals[v * 3 + 1]!, tube.normals[v * 3 + 2]!),
                                            tangent: new float4(tube.tangents[v * 4]!, tube.tangents[v * 4 + 1]!, tube.tangents[v * 4 + 2]!, tube.tangents[v * 4 + 3]!),
                                            texCrd: new float2(0, 0),
                                            curveRadius: tube.radii[v]!,
                                        });
                                    }
                                    meshes.push({ vertices, indices: tube.faceVertexIndices, materialID });
                                    continue;
                                }
                                const r = convertToLinearSweptSphere(strandCount, strand.curveVertexCounts, strand.points, strand.widths, null, 1, subdiv, keepStrands, keepVertices, widthScale, float4x4.identity());
                                const positionsRadii = new Float32Array(r.points.length * 4);
                                r.points.forEach((pnt, vi) => positionsRadii.set([pnt.x, pnt.y, pnt.z, r.radius[vi]!], vi * 4));
                                curves.push({ positionsRadii, texCrds: null, indices: r.indices, materialID });
                            }
                        }
                    }
                } else if (kAssimpSceneExtensions.includes(cmd.path.slice(cmd.path.lastIndexOf(".") + 1).toLowerCase())) {
                    // Every format AssimpImporter registers except glTF/USD/pbrt,
                    // which have their own importers (as natively).
                    const dir = url.slice(0, url.lastIndexOf("/"));
                    const fileName = cmd.path.slice(cmd.path.lastIndexOf("/") + 1);
                    // OBJ materials live in side files that assimp opens by name.
                    const extraFiles: { name: string; bytes: Uint8Array }[] = [];
                    if (fileName.toLowerCase().endsWith(".obj")) {
                        for (const lib of objMaterialLibraries(new TextDecoder().decode(bytes))) {
                            const res = await fetch(`${dir}/${lib}`);
                            if (res.ok) extraFiles.push({ name: lib, bytes: new Uint8Array(await res.arrayBuffer()) });
                            else Logger.warning(`AssimpImporter: material library '${lib}' not found next to '${fileName}'.`);
                        }
                    }
                    const parsed = await FbxImporter.parseToDescs(bytes, dir, textureManager, {
                        ...this.importOptions,
                        useSpecGloss: this.hasFlag(SceneBuilderFlags.UseSpecGlossMaterials),
                        useMetalRough: this.hasFlag(SceneBuilderFlags.UseMetalRoughMaterials),
                        fileName,
                        extraFiles,
                    });
                    parsed.materials.forEach((m, i) => (m.name ??= parsed.materialNames[i]));
                    materials.push(...parsed.materials);
                    importedMaterialNames.push(...parsed.materialNames);
                    const nodeOffset = nodes.length;
                    for (const n of parsed.nodes) nodes.push({ ...n, parent: n.parent >= 0 ? n.parent + nodeOffset : -1 });
                    for (const ch of parsed.animations)
                        animations.push({ ...ch, nodeID: ch.nodeID + nodeOffset, clip: ch.clip !== undefined ? ch.clip + clipOffset : undefined });
                    clipOffset += parsed.animations.reduce((mx, c) => Math.max(mx, (c.clip ?? -1) + 1), 0);
                    importedLights.push(...parsed.lights);
                    for (const m of parsed.meshes)
                        meshes.push({
                            ...m,
                            materialID: m.materialID + materialOffset,
                            nodeID: m.nodeID !== undefined ? m.nodeID + nodeOffset : undefined,
                            skin: m.skin ? { ...m.skin, boneNodeIDs: m.skin.boneNodeIDs.map((n) => n + nodeOffset) } : undefined,
                        });
                } else {
                    const parsed = await GltfImporter.parseToDescs(bytes, url, textureManager, this.importOptions);
                    materials.push(...parsed.materials);
                    importedMaterialNames.push(...parsed.materials.map(() => ""));
                    // Offset the imported node graph so multiple imports don't collide.
                    const nodeOffset = nodes.length;
                    for (const n of parsed.nodes) nodes.push({ ...n, parent: n.parent >= 0 ? n.parent + nodeOffset : -1 });
                    for (const ch of parsed.animations) animations.push({ ...ch, nodeID: ch.nodeID + nodeOffset });
                    for (const l of parsed.lights) importedLights.push({ ...l, nodeID: l.nodeID !== undefined ? l.nodeID + nodeOffset : undefined });
                    for (const wt of parsed.weightTracks) weightTracks.push({ ...wt, nodeID: wt.nodeID + nodeOffset });
                    if (parsed.cameraNodeID !== undefined && this.importedCameraNodeID === undefined) {
                        this.importedCameraNodeID = parsed.cameraNodeID + nodeOffset;
                        this.importedCameraPose = parsed.camera;
                    }
                    for (const m of parsed.meshes)
                        meshes.push({
                            ...m,
                            materialID: m.materialID + materialOffset,
                            nodeID: m.nodeID !== undefined ? m.nodeID + nodeOffset : undefined,
                            skin: m.skin ? { ...m.skin, boneNodeIDs: m.skin.boneNodeIDs.map((n) => n + nodeOffset) } : undefined,
                            morph: m.morph ? { ...m.morph, nodeID: m.morph.nodeID + nodeOffset } : undefined,
                        });
                }
            }
        }

        // Apply the recorded pyscene pre/post-infinity behavior writes to the
        // imported clips (mirrors sceneBuilder.animations[i].preInfinityBehavior).
        for (const [i, h] of this.animationHandles.entries()) {
            const pre = h.preInfinityBehavior;
            const post = h.postInfinityBehavior;
            if (pre == null && post == null) continue;
            for (const ch of animations) {
                if (ch.clip !== i) continue;
                if (pre != null) ch.preInfinity = Number(pre);
                if (post != null) ch.postInfinity = Number(post);
            }
        }

        // Apply deferred getMaterial() edits (mirrors pyscene mutations after importScene).
        for (const edit of this.materialEdits) {
            const idx = importedMaterialNames.indexOf(edit.name);
            if (idx < 0) throw new RuntimeError(`SceneBuilder.getMaterial: unknown material '${edit.name}'`);
            const mat = materials[idx]!;
            if (!applyMaterialEdit(mat, edit.prop, edit.value)) {
                Logger.warning(`SceneBuilder.getMaterial('${edit.name}').${edit.prop}: unsupported on the web bridge; ignored`);
            }
        }

        // Global emissive boost (see globalEmissiveScale) — scales every imported
        // material's emissive contribution so interior emissives read as lights.
        if (this.globalEmissiveScale !== 1) {
            for (const mat of materials) mat.basic.emissiveFactor = (mat.basic.emissiveFactor ?? 1) * this.globalEmissiveScale;
        }

        // Resolve TriangleMesh.createFromFile() geometry (deferred async asset load).
        for (const geo of this.meshGeometry) {
            if (!geo._fromFile) continue;
            const url = await resolveAssetUrl(geo._fromFile.path, baseUrl, AssetCategory.Any, this.assetResolver);
            const res = await fetch(url);
            if (!res.ok) throw new RuntimeError(`TriangleMesh.createFromFile: failed to fetch '${url}' (${res.status})`);
            const loaded = await FbxImporter.parseMeshOnly(new Uint8Array(await res.arrayBuffer()), geo._fromFile.path, geo._fromFile.smoothNormals);
            geo.vertices = loaded.vertices;
            geo.indices = loaded.indices;
            geo._fromFile = undefined;
        }

        // Load deferred material textures (material.loadTexture()).
        for (const mat of new Set(this.meshMaterials)) {
            await mat.resolveTextures(baseUrl, textureManager, this.assetResolver, this.hasFlag(SceneBuilderFlags.AssumeLinearSpaceTextures));
            await mat.resolveMeasured(baseUrl, this.assetResolver);
        }

        // Builder-added meshes (instanced via nodes).
        const materialIDs = new Map<MaterialBridge, number>();
        this.meshGeometry.forEach((geo, meshID) => {
            const transforms = this.meshInstanced.get(meshID);
            if (!transforms) return; // mesh never instanced
            const mat = this.meshMaterials[meshID]!;
            let materialID = materialIDs.get(mat);
            if (materialID === undefined) {
                materialID = materials.length;
                materials.push(mat.toDesc());
                materialIDs.set(mat, materialID);
            }
            // Local-space tangents when the asset provides none (native MikkTSpace);
            // UseOriginalTangentSpace keeps supplied tangents.
            const vertices = geo.vertices.map((v) => ({ ...v }));
            const hasTangents = vertices.some((v) => v.tangent.x !== 0 || v.tangent.y !== 0 || v.tangent.z !== 0);
            if (!(this.hasFlag(SceneBuilderFlags.UseOriginalTangentSpace) && hasTangents)) generateTangents(vertices, geo.indices);
            for (const transform of transforms) {
                meshes.push({ vertices, indices: geo.indices, materialID, transform });
            }
        });

        // Builder-added curves share the builder materials' IDs.
        for (const c of this.builderCurves) {
            let materialID = materialIDs.get(c.material);
            if (materialID === undefined) {
                materialID = materials.length;
                materials.push(c.material.toDesc());
                materialIDs.set(c.material, materialID);
            }
            curves.push({ positionsRadii: c.positionsRadii, texCrds: null, indices: c.indices, materialID, transform: this.nodes[c.nodeID]! });
        }

        const lights: AnalyticLight[] = this.lights.map((l) => {
            const isArea = l.lightType === LightType.Rect || l.lightType === LightType.Disc || l.lightType === LightType.Sphere;
            return {
                type: l.lightType,
                name: l.name,
                posW: l.getPosition(),
                dirW: l.getDirection(),
                intensity: l.getIntensity(),
                angle: l.angle,
                openingAngle: l.lightType === LightType.Point ? l.openingAngle : undefined,
                penumbraAngle: l.penumbraAngle,
                transMat: isArea ? l.getTransMat() : undefined,
            };
        });
        lights.push(...importedLights); // lights imported from FBX/assets

        // SDF grids (ND + SBS implementations; instances reference builder nodes).
        const sdfGrids: SceneSDFGridDesc[] = [];
        // Fetch any `.sdfg` corner values first; they become ops on the recipe.
        for (const { grid } of this.sdfGridsList) {
            for (const file of grid.pendingFiles) {
                const url = await resolveAssetUrl(file.path, baseUrl, AssetCategory.Any, this.assetResolver);
                if (file.primitives) {
                    const { loadSDFPrimitives } = await import("./SDFs/SDF3DPrimitive.js");
                    grid.ops.push({ kind: "primitives", gridWidth: file.gridWidth!, primitives: await loadSDFPrimitives(url) });
                    continue;
                }
                const { loadSDFGridValues } = await import("./SDFs/SDFGridFile.js");
                const loaded = await loadSDFGridValues(url);
                grid.ops.push({ kind: "values", gridWidth: loaded.gridWidth, values: loaded.values });
            }
            grid.pendingFiles = [];
        }
        const sdfRecipes = this.sdfGridsList.map(({ grid }) => grid.toRecipe());
        const builtSdfGrids = this.sdfGridsList.map(({ material }, i) => {
            const built = buildSDFGridFromRecipe(sdfRecipes[i]!);
            let materialID = materialIDs.get(material);
            if (materialID === undefined) {
                materialID = materials.length;
                materials.push(material.toDesc());
                materialIDs.set(material, materialID);
            }
            return { grid: built, materialID };
        });
        for (const inst of this.sdfInstances) {
            const built = builtSdfGrids[inst.sdfGridID];
            if (!built) throw new RuntimeError(`addSDFGridInstance: unknown SDF grid ${inst.sdfGridID}`);
            sdfGrids.push({ grid: built.grid, materialID: built.materialID, transform: this.nodes[inst.nodeID]! });
        }

        // Flags::DontUseDisplacement: drop displacement maps (meshes stay plain triangles).
        if (this.hasFlag(SceneBuilderFlags.DontUseDisplacement)) for (const m of materials) delete m.basic.texDisplacement;
        // Flags::NonIndexedVertices: every triangle gets its own vertices.
        if (this.hasFlag(SceneBuilderFlags.NonIndexedVertices)) meshes.forEach((m, i) => (meshes[i] = deindexMesh(m)));

        // Only bind the camera to an imported node when the pyscene doesn't define
        // its own camera (an explicit pyscene camera wins and stays static).
        const cameraNodeID = this.camera ? undefined : this.importedCameraNodeID;
        const scene = new Scene(device, meshes, materials, lights, textureManager, sdfGrids, nodes, animations, cameraNodeID, weightTracks, curves);
        for (const c of this.customPrimitives) scene.addCustomPrimitive(c.userID, c.aabb);
        // Snapshot for the scene cache (v4: every scene class; grid volumes are
        // read off scene.gridVolumes after finalize, env map off the scene).
        this.lastSceneArgs = {
            meshes,
            materials,
            lights,
            nodes,
            cameraNodeID,
            textureManager,
            curves,
            animations,
            weightTracks,
            sdfGrids: {
                recipes: sdfRecipes,
                instances: this.sdfInstances.map((inst) => ({ gridIndex: inst.sdfGridID, materialID: builtSdfGrids[inst.sdfGridID]!.materialID, transform: this.nodes[inst.nodeID] })),
            },
            cacheable: true,
        };
        if (this.camera) {
            scene.camera.setPosition(this.camera.getPosition());
            scene.camera.setTarget(this.camera.getTarget());
            scene.camera.setUpVector(this.camera.getUp());
            scene.camera.setFocalLength(this.camera.focalLength);
            scene.camera.setFocalDistance(this.camera.focalDistance);
            scene.camera.setApertureRadius(this.camera.apertureRadius);
            scene.camera.setShutterSpeed(this.camera.shutterSpeed);
            scene.camera.setISOSpeed(this.camera.ISOSpeed);
            scene.camera.setDepthRange(this.camera.nearPlane, this.camera.farPlane);
        } else if (this.importedCameraPose) {
            scene.camera.setPosition(this.importedCameraPose.position);
            scene.camera.setTarget(this.importedCameraPose.target);
            scene.camera.setUpVector(this.importedCameraPose.up);
            scene.camera.setFocalLength(this.importedCameraPose.focalLength);
        }
        if (this.envMap) {
            const constant = this.envMap.constantColor;
            const envMap = constant
                ? new EnvMap(device, { width: 1, height: 1, data: new Float32Array([constant[0], constant[1], constant[2], 1]) })
                : await EnvMap.createFromUrl(device, await resolveAssetUrl(this.envMap.path, baseUrl, AssetCategory.Any, this.assetResolver), {
                      equalAreaOctahedral: this.envMap.equalAreaOctahedral,
                  });
            envMap.intensity = this.envMap.intensity;
            if (this.envMap.rotation) envMap.setRotation([this.envMap.rotation.x, this.envMap.rotation.y, this.envMap.rotation.z]);
            scene.setEnvMap(envMap);
        }
        for (const v of this.gridVolumesList) {
            const vol = new GridVolume(v.name);
            vol.densityScale = v.densityScale;
            vol.emissionScale = v.emissionScale;
            vol.albedo = new float3(v.albedo.x, v.albedo.y, v.albedo.z);
            vol.anisotropy = v.anisotropy;
            vol.emissionTemperature = v.emissionTemperature;
            for (const g of v.grids) {
                const url = await resolveAssetUrl(g.path, baseUrl, AssetCategory.Any, this.assetResolver);
                vol.setGrid(g.slot as GridSlot, await Grid.createFromUrl(device, url, g.gridname));
            }
            for (const seq of v.gridSequences) {
                const grids = [];
                for (const path of seq.paths) {
                    const url = await resolveAssetUrl(path, baseUrl, AssetCategory.Any, this.assetResolver);
                    grids.push(await Grid.createFromUrl(device, url, seq.gridname));
                }
                vol.setGridSequence(seq.slot as GridSlot, grids);
            }
            vol.frameRate = v.frameRate;
            vol.playbackEnabled = v.playbackEnabled;
            vol.startFrame = v.startFrame;
            for (const pg of v.proceduralGrids) {
                vol.setGrid(pg.slot as GridSlot, new Grid(device, buildNanoVDBGrid(pg.parsed)));
            }
            scene.gridVolumes.push(vol);
        }
        scene.finalizeGridVolumes();

        // Mirrors MaterialSystem::update's deferred bake of the IES profile.
        if (this.lightProfileRequest) {
            const { LightProfile } = await import("./Lights/LightProfile.js");
            const url = await resolveAssetUrl(this.lightProfileRequest.path, baseUrl, AssetCategory.Any, this.assetResolver);
            const profile = await LightProfile.createFromIesProfile(device, url, this.lightProfileRequest.normalize);
            await profile.bake(device.renderContext);
            scene.lightProfile = profile;
        }
        return scene;
    }
}
