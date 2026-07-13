/**
 * SceneBuilder bridge for .pyscene execution (docs §11.1). Builder calls record
 * commands synchronously; resolve() then fetches assets and constructs the Scene.
 */

import type { Device } from "../Core/API/Device.js";
import { Grid } from "./Volume/Grid.js";
import { GridVolume, type GridSlot } from "./Volume/GridVolume.js";
import { buildNanoVDBGrid, type ParsedFloatGrid } from "./Volume/VDBLoader.js";
import { NDSDFGrid } from "./SDFs/NDSDFGrid.js";
import { SDFSBS } from "./SDFs/SDFSBS.js";
import { SDFSVS } from "./SDFs/SDFSVS.js";
import { SDFSVO } from "./SDFs/SDFSVO.js";
import type { SceneSDFGridDesc } from "./Scene.js";
import { Scene, type SceneMaterialDesc, type SceneMeshDesc } from "./Scene.js";
import type { SceneNode, AnimationChannel, WeightTrack } from "./Animation/SceneAnimation.js";
import { GltfImporter } from "./Importer/GltfImporter.js";
import { FbxImporter } from "./Importer/FbxImporter.js";
import { UsdImporter } from "./Importer/UsdImporter.js";
import { convertToLinearSweptSphere, extractBasisCurvesFromUsda } from "./Curves/CurveTessellation.js";
import { TextureManager } from "./Material/TextureManager.js";
import { EnvMap } from "./Lights/EnvMap.js";
import { generateTangents } from "./TangentSpace.js";
import { LightType, type AnalyticLight, type StaticVertex } from "./SceneData.js";
import { MaterialType, packTextureHandle, TextureHandleMode } from "./Material/MaterialData.js";
import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { float4x4, matrixFromTranslation, matrixFromScaling, mulMat } from "../Utils/Math/Matrix.js";
import { RuntimeError } from "../Core/Error.js";
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
    ) {}

    // Deferred texture loads (material.loadTexture(slot, path)); resolved in resolve().
    private _textures: { slot: string; path: string }[] = [];
    private _texHandles: { texBaseColor?: number; texSpecular?: number; texEmissive?: number; texNormalMap?: number; texDisplacement?: number } = {};

    loadTexture(slot: string, path: string): void {
        this._textures.push({ slot: String(slot), path: String(path) });
    }

    /** Fetches + decodes this material's deferred textures into the TextureManager. */
    async resolveTextures(baseUrl: string, tm: TextureManager): Promise<void> {
        for (const t of this._textures) {
            const url = baseUrl ? `${baseUrl}/${t.path}` : t.path;
            try {
                const res = await fetch(url);
                if (!res.ok) continue;
                const srgb = t.slot === "BaseColor" || t.slot === "Emissive";
                const blob = await res.blob();
                const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
                const bytes = new Uint8Array(await blob.arrayBuffer());
                const handle = packTextureHandle(TextureHandleMode.Texture, tm.addTexture({ bitmap, srgb, bytes }));
                if (t.slot === "BaseColor") this._texHandles.texBaseColor = handle;
                else if (t.slot === "Specular") this._texHandles.texSpecular = handle;
                else if (t.slot === "Normal") this._texHandles.texNormalMap = handle;
                else if (t.slot === "Emissive") this._texHandles.texEmissive = handle;
                else if (t.slot === "Displacement") this._texHandles.texDisplacement = handle;
            } catch {
                /* undecodable format (e.g. DDS) — material falls back to base color */
            }
        }
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
        if (typeof r === "number") {
            this._specularParams = new float4(this._specularParams.x, r, this._specularParams.z, this._specularParams.w);
        } else {
            // PBRTConductorMaterial::setRoughness(float2) -> specular.rg.
            this._specularParams = new float4(r.x, r.y, this._specularParams.z, this._specularParams.w);
        }
    }

    /** StandardMaterial::setMetallic -> specular.b. */
    set metallic(m: number) {
        this._specularParams = new float4(this._specularParams.x, this._specularParams.y, m, this._specularParams.w);
    }

    toDesc(): SceneMaterialDesc {
        const emissive = this._emissiveColor.x !== 0 || this._emissiveColor.y !== 0 || this._emissiveColor.z !== 0;
        return {
            name: this.name,
            header: {
                materialType: this.materialType,
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

export type SDFGridType = "ndsdf" | "sbs" | "svs" | "svo";

/** Recorded SDF grid state (mirrors SDFGrid python bindings; ND + SBS types). */
export class SDFGridBridge {
    ops: { kind: "cheese"; gridWidth: number; seed: number }[] = [];
    constructor(
        readonly type: SDFGridType,
        readonly narrowBandThickness: number,
        readonly brickWidth: number,
    ) {}
    generateCheeseValues(gridWidth: number, seed: number): void {
        this.ops.push({ kind: "cheese", gridWidth: Number(gridWidth), seed: Number(seed) });
    }
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
    proceduralGrids: { slot: string; parsed: ParsedFloatGrid }[] = [];

    constructor(name = "") {
        this.name = String(name);
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
    path: string;
    intensity: number;
    rotation?: { x: number; y: number; z: number };
}

type Command =
    | { kind: "import"; path: string }
    | { kind: "mesh"; mesh: TriangleMeshDesc; material: MaterialBridge; nodeTransform: float4x4 };

export class SceneBuilderBridge {
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

    /** Deferred animation handles (imports resolve later; web animation loops the
     *  whole clip, so pre/post-infinity behavior writes are accepted and ignored). */
    get animations(): { preInfinityBehavior: unknown; postInfinityBehavior: unknown }[] {
        return Array.from({ length: 16 }, () => ({ preInfinityBehavior: null, postInfinityBehavior: null }));
    }

    importScene(path: string): void {
        this.commands.push({ kind: "import", path });
    }

    addTriangleMesh(mesh: TriangleMeshDesc, material: MaterialBridge): number {
        this.meshGeometry.push(mesh);
        this.meshMaterials.push(unwrapGuard(material));
        return this.meshGeometry.length - 1;
    }

    addNode(_name: string, transform: float4x4): number {
        this.nodes.push(transform);
        return this.nodes.length - 1;
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

    /** Renders a custom primitive as its AABB box (web approximation: Falcor's
     *  procedural custom primitives need an app-supplied intersection shader, which
     *  the software ray tracer has no equivalent for). */
    addCustomPrimitive(userID: number, aabb: { min: { x: number; y: number; z: number }; max: { x: number; y: number; z: number } }): void {
        const { min, max } = aabb;
        const size = new float3(max.x - min.x, max.y - min.y, max.z - min.z);
        const center = new float3((max.x + min.x) / 2, (max.y + min.y) / 2, (max.z + min.z) / 2);
        const id = Number(userID);
        const palette = [
            [0.9, 0.3, 0.3],
            [0.3, 0.9, 0.4],
            [0.3, 0.5, 0.9],
            [0.9, 0.8, 0.2],
        ];
        const c = palette[id % palette.length]!;
        const mat = new MaterialBridge(MaterialType.Standard, `CustomPrimitive_${id}`);
        mat.baseColor = { x: c[0]!, y: c[1]!, z: c[2]!, w: 1 };
        mat.roughness = 0.5;
        const meshID = this.addTriangleMesh(TriangleMesh.createCube(size), mat);
        this.addMeshInstance(this.addNode("", matrixFromTranslation(center)), meshID);
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

    addSDFGrid(grid: SDFGridBridge, material: MaterialBridge): number {
        const g = unwrapGuard(grid) as SDFGridBridge;
        const copy = new SDFGridBridge(g.type, Number(g.narrowBandThickness), Number(g.brickWidth));
        copy.ops = g.ops.map((o) => ({ ...o }));
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
        for (const cmd of this.commands) {
            if (cmd.kind === "import") {
                const url = baseUrl ? `${baseUrl}/${cmd.path}` : cmd.path;
                const res = await fetch(url);
                if (!res.ok) throw new RuntimeError(`SceneBuilder: failed to fetch '${url}' (${res.status})`);
                const bytes = new Uint8Array(await res.arrayBuffer());
                const materialOffset = materials.length;
                if (/\.usd[acz]?$/.test(cmd.path.toLowerCase())) {
                    const dir = url.slice(0, url.lastIndexOf("/"));
                    // Curve prims import as real curves below; tinyusdz also
                    // tessellates them into meshes — exclude those duplicates.
                    const curvePrims = cmd.path.toLowerCase().endsWith(".usda")
                        ? extractBasisCurvesFromUsda(new TextDecoder().decode(bytes))
                        : [];
                    const parsed = await UsdImporter.parseToDescs(bytes, textureManager, dir, new Set(curvePrims.map((c) => c.name)));
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
                            for (const strand of strands) {
                                const r = convertToLinearSweptSphere(strand.curveVertexCounts.length, strand.curveVertexCounts, strand.points, strand.widths, null, 1, 1, 1, 1, 1, float4x4.identity());
                                const positionsRadii = new Float32Array(r.points.length * 4);
                                r.points.forEach((pnt, vi) => positionsRadii.set([pnt.x, pnt.y, pnt.z, r.radius[vi]!], vi * 4));
                                curves.push({ positionsRadii, texCrds: null, indices: r.indices, materialID });
                            }
                        }
                    }
                } else if (cmd.path.toLowerCase().endsWith(".fbx")) {
                    const dir = url.slice(0, url.lastIndexOf("/"));
                    const parsed = await FbxImporter.parseToDescs(bytes, dir, textureManager);
                    parsed.materials.forEach((m, i) => (m.name ??= parsed.materialNames[i]));
                    materials.push(...parsed.materials);
                    importedMaterialNames.push(...parsed.materialNames);
                    const nodeOffset = nodes.length;
                    for (const n of parsed.nodes) nodes.push({ ...n, parent: n.parent >= 0 ? n.parent + nodeOffset : -1 });
                    for (const ch of parsed.animations) animations.push({ ...ch, nodeID: ch.nodeID + nodeOffset });
                    importedLights.push(...parsed.lights);
                    for (const m of parsed.meshes)
                        meshes.push({
                            ...m,
                            materialID: m.materialID + materialOffset,
                            nodeID: m.nodeID !== undefined ? m.nodeID + nodeOffset : undefined,
                            skin: m.skin ? { ...m.skin, boneNodeIDs: m.skin.boneNodeIDs.map((n) => n + nodeOffset) } : undefined,
                        });
                } else {
                    const parsed = await GltfImporter.parseToDescs(bytes, url, textureManager);
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
            const url = baseUrl ? `${baseUrl}/${geo._fromFile.path}` : geo._fromFile.path;
            const res = await fetch(url);
            if (!res.ok) throw new RuntimeError(`TriangleMesh.createFromFile: failed to fetch '${url}' (${res.status})`);
            const loaded = await FbxImporter.parseMeshOnly(new Uint8Array(await res.arrayBuffer()), geo._fromFile.path);
            geo.vertices = loaded.vertices;
            geo.indices = loaded.indices;
            geo._fromFile = undefined;
        }

        // Load deferred material textures (material.loadTexture()).
        for (const mat of new Set(this.meshMaterials)) await mat.resolveTextures(baseUrl, textureManager);

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
            // Local-space tangents when the asset provides none (native MikkTSpace).
            const vertices = geo.vertices.map((v) => ({ ...v }));
            generateTangents(vertices, geo.indices);
            for (const transform of transforms) {
                meshes.push({ vertices, indices: geo.indices, materialID, transform });
            }
        });

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
        const builtSdfGrids = this.sdfGridsList.map(({ grid, material }) => {
            const built: NDSDFGrid | SDFSBS | SDFSVS | SDFSVO =
                grid.type === "sbs" ? new SDFSBS(grid.brickWidth) : grid.type === "svs" ? new SDFSVS() : grid.type === "svo" ? new SDFSVO() : new NDSDFGrid(grid.narrowBandThickness);
            for (const op of grid.ops) {
                if (op.kind === "cheese") built.generateCheeseValues(op.gridWidth, op.seed);
            }
            const built_ok = built instanceof SDFSBS ? built.brickCount > 0 : built instanceof SDFSVS || built instanceof SDFSVO ? built.voxelCount > 0 : built.lodCount > 0;
            if (!built_ok) throw new RuntimeError("SDFGrid: no values set (only generateCheeseValues is supported so far)");
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

        // Only bind the camera to an imported node when the pyscene doesn't define
        // its own camera (an explicit pyscene camera wins and stays static).
        const cameraNodeID = this.camera ? undefined : this.importedCameraNodeID;
        const scene = new Scene(device, meshes, materials, lights, textureManager, sdfGrids, nodes, animations, cameraNodeID, weightTracks, curves);
        // Snapshot for the scene cache (phase 1: static texture-less scenes only).
        this.lastSceneArgs = {
            meshes,
            materials,
            lights,
            nodes,
            cameraNodeID,
            textureManager,
            cacheable:
                animations.length === 0 &&
                weightTracks.length === 0 &&
                curves.length === 0 &&
                sdfGrids.length === 0 &&
                !this.envMap &&
                this.gridVolumesList.length === 0 &&
                meshes.every((m) => !m.skin && !m.morph),
        };
        if (this.camera) {
            scene.camera.setPosition(this.camera.getPosition());
            scene.camera.setTarget(this.camera.getTarget());
            scene.camera.setUpVector(this.camera.getUp());
            scene.camera.setFocalLength(this.camera.focalLength);
            scene.camera.setFocalDistance(this.camera.focalDistance);
            scene.camera.setApertureRadius(this.camera.apertureRadius);
        } else if (this.importedCameraPose) {
            scene.camera.setPosition(this.importedCameraPose.position);
            scene.camera.setTarget(this.importedCameraPose.target);
            scene.camera.setUpVector(this.importedCameraPose.up);
            scene.camera.setFocalLength(this.importedCameraPose.focalLength);
        }
        if (this.envMap) {
            const url = baseUrl ? `${baseUrl}/${this.envMap.path}` : this.envMap.path;
            const envMap = await EnvMap.createFromUrl(device, url);
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
                const url = baseUrl ? `${baseUrl}/${g.path}` : g.path;
                vol.setGrid(g.slot as GridSlot, await Grid.createFromUrl(device, url, g.gridname));
            }
            for (const pg of v.proceduralGrids) {
                vol.setGrid(pg.slot as GridSlot, new Grid(device, buildNanoVDBGrid(pg.parsed)));
            }
            scene.gridVolumes.push(vol);
        }
        scene.finalizeGridVolumes();
        return scene;
    }
}
