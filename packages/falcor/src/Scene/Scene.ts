/**
 * Scene host class mirroring Falcor/Scene/Scene.h (v1: static triangle meshes,
 * basic materials, single camera; grows toward full parity per milestone).
 *
 * Owns the GPU buffers laid out per SceneTypes.slang and binds the gScene
 * parameter block (upstream Scene.slang via WebFalcor overrides).
 */

import { ScriptWriter } from "../Utils/Scripting/ScriptWriter.js";
import type { Device } from "../Core/API/Device.js";
import { Buffer } from "../Core/API/Buffer.js";
import { Texture } from "../Core/API/Texture.js";
import { Sampler, TextureFilteringMode, TextureAddressingMode } from "../Core/API/Sampler.js";
import { ResourceBindFlags, MemoryType, ResourceType } from "../Core/API/Types.js";
import { ResourceFormat } from "../Core/API/Formats.js";
import { DefineList } from "../Core/Program/DefineList.js";
import type { ShaderVar } from "../Core/Program/ParameterBlock.js";
import { Camera } from "./Camera/Camera.js";
import { float4x4, transpose, inverse } from "../Utils/Math/Matrix.js";
import { buildBvh, buildBvhParallel, buildAabbBvh, refitBvh, type BvhBuildResult, type BvhTriangle } from "./SoftwareRT/Bvh.js";
import { WorkerPool } from "../Utils/Threading/WorkerPool.js";
import { packLights, LightType, SceneLight, type AnalyticLight } from "./SceneData.js";
import { TextureManager, kMaxTextureBuckets } from "./Material/TextureManager.js";
import type { EnvMap } from "./Lights/EnvMap.js";
import { buildLightCollection } from "./Lights/LightCollection.js";
import { evaluateGlobals, computeSkinMatrices, skinVertices, sampleMorphWeights, applyMorph, type SceneAnimations, type SceneNode, type AnimationChannel, type SkinDesc, type MorphDesc, type WeightTrack } from "./Animation/SceneAnimation.js";
import { decodeNormal2x16Host, type Vec3 } from "../Rendering/Lights/LightBVHTypes.js";
import type { EmissiveTriangleInput } from "../Rendering/Lights/LightBVHBuilder.js";
import { transformPoint, transformVector } from "../Utils/Math/Matrix.js";
import { float2, float3, float4, normalize3 } from "../Utils/Math/Vector.js";
import { quatf, rotateVector } from "../Utils/Math/Quaternion.js";
import { float16ToFloat32, float32ToFloat16 } from "../Utils/Math/Float16.js";
import {
    GeometryType,
    packGeometryInstances,
    packMeshDescs,
    packStaticVertices,
    packPrevVertices,
    type GeometryInstance,
    type MeshDescData,
    type StaticVertex,
} from "./SceneData.js";
import { packBasicMaterialBlob, packMERLMaterialBlob, packMERLMixMaterialBlob, packRGLMaterialBlob, writeDiffuseSpecularData, kDiffuseSpecularDataSize, AlphaMode, MaterialType, ShadingModel, TextureHandleMode, type BasicMaterialDesc, type MaterialHeaderDesc } from "./Material/MaterialData.js";
import { kMERLAlbedoLUTSize, type MERLBRDF, type MERLMixData } from "./Material/MERLFile.js";
import { kRGLAlbedoLUTSize, type RGLMeasurement } from "./Material/RGLFile.js";
import type { LightProfile } from "./Lights/LightProfile.js";
import type { RenderContext } from "../Core/API/RenderContext.js";
import { assert, RuntimeError } from "../Core/Error.js";
import { formatByteSize } from "../Utils/StringUtils.js";
import { Logger } from "../Utils/Logger.js";
import { SceneMaterial } from "./Material/SceneMaterial.js";
import { AABB } from "../Utils/Math/AABB.js";
import { Rectangle } from "../Utils/Math/Rectangle.js";
import { AssetCategory, resolveAssetUrl } from "../Core/AssetResolver.js";
import { getFormatChannelCount } from "../Core/API/Formats.js";
import type { NDSDFGrid } from "./SDFs/NDSDFGrid.js";
import { SDFSBS, packSBSGrids, type PackedSBS } from "./SDFs/SDFSBS.js";
import { encodeBC4Texture } from "./SDFs/BC4Encode.js";
import { SDFSVS } from "./SDFs/SDFSVS.js";
import { SDFSVO } from "./SDFs/SDFSVO.js";

/** One SDF grid instance (mirrors Scene::mSDFGrids + mSDFGridDesc + instance). */
export interface SceneSDFGridDesc {
    grid: NDSDFGrid | SDFSBS | SDFSVS | SDFSVO;
    materialID: number;
    transform?: float4x4;
}

/** Vertex arrays already rounded to f32 (Scene.create rounds before the constructor). */
const roundedVertexArrays = new WeakSet<StaticVertex[]>();

/** Segment-AABB BVH over curves (prim entries encode instance << 24 | segment), plus its node word count. */
function buildCurveBvh(curves: SceneCurveDesc[]): { data: Float32Array; nodeWords: number } {
    const segAabbs: { min: [number, number, number]; max: [number, number, number] }[] = [];
    curves.forEach((curve) => {
        const m = curve.transform ?? float4x4.identity();
        const scale = Math.hypot(m.get(0, 0), m.get(0, 1), m.get(0, 2));
        for (const seg of curve.indices) {
            const lo: [number, number, number] = [Infinity, Infinity, Infinity];
            const hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
            for (const v of [seg, seg + 1]) {
                const p = transformPoint(m, new float3(curve.positionsRadii[v * 4]!, curve.positionsRadii[v * 4 + 1]!, curve.positionsRadii[v * 4 + 2]!));
                const r = curve.positionsRadii[v * 4 + 3]! * scale;
                for (const [a, c] of [[0, p.x], [1, p.y], [2, p.z]] as const) {
                    lo[a] = Math.min(lo[a], c - r);
                    hi[a] = Math.max(hi[a], c + r);
                }
            }
            segAabbs.push({ min: lo, max: hi });
        }
    });
    const curveBvh = buildAabbBvh(segAabbs);
    const encoded = new Uint32Array(curveBvh.primIndices.length);
    const segToEnc: number[] = [];
    curves.forEach((curve, ci) => {
        for (let sIdx = 0; sIdx < curve.indices.length; sIdx++) segToEnc.push(((ci & 0xff) << 24) | sIdx);
    });
    for (let i = 0; i < curveBvh.primIndices.length; i++) encoded[i] = segToEnc[curveBvh.primIndices[i]!] ?? 0;
    const primWords = Math.ceil(encoded.length / 4) * 4;
    const data = new Float32Array(curveBvh.nodes.length + primWords);
    data.set(curveBvh.nodes, 0);
    new Uint32Array(data.buffer, curveBvh.nodes.length * 4).set(encoded);
    return { data, nodeWords: curveBvh.nodes.length };
}

/** Curve vertex/index/metadata buffers. */
function packCurves(curves: SceneCurveDesc[]): { cv: Float32Array; ci: Uint32Array; cd: Uint32Array } {
            // StaticCurveVertexData WGSL layout: position@0, radius@12, texCrd@16, stride 32.
    const totalVerts = curves.reduce((acc, c) => acc + c.positionsRadii.length / 4, 0);
    const cv = new Float32Array(totalVerts * 8);
    const totalSegs = curves.reduce((acc, c) => acc + c.indices.length, 0);
    const ci = new Uint32Array(totalSegs);
    const cd = new Uint32Array(curves.length * 6);
    let vtx = 0;
    let seg = 0;
    curves.forEach((curve, i) => {
        const count = curve.positionsRadii.length / 4;
        for (let v = 0; v < count; v++) {
            cv.set(curve.positionsRadii.subarray(v * 4, v * 4 + 4), (vtx + v) * 8);
            if (curve.texCrds) cv.set(curve.texCrds.subarray(v * 2, v * 2 + 2), (vtx + v) * 8 + 4);
        }
        ci.set(curve.indices, seg);
        cd.set([vtx, seg, count, curve.indices.length, 1, curve.materialID], i * 6);
        vtx += count;
        seg += curve.indices.length;
    });
    return { cv, ci, cd };
}

/** calculateInterpolation for curve caches (post-infinity Constant): keyframes a, b and fraction t. */
function curveInterpolation(ts: number[], time: number, preCycle: boolean): [number, number, number] {
    let [a, b, t] = [0, 0, 0];
    time = Math.max(time, 0);
    if (time > ts[ts.length - 1]!) [a, b] = [ts.length - 1, ts.length - 1];
    else if (time <= ts[0]!) {
        if (preCycle) [a, b, t] = [ts.length - 1, 0, time / ts[0]!];
    } else {
        b = ts.findIndex((x) => x >= time);
        a = b - 1;
        t = (time - ts[a]!) / (ts[b]! - ts[a]!);
    }
    return [a, b, t];
}

/**
 * UpdateCurvePolyTubeVertices: each tube vertex's ring center follows the interpolated curve, its
 * normal turns by the rotation from its tangent to the curve's (forward difference, backward at a
 * strand's end) and it sits at center + radius * normal. Native applies this frame to frame; after
 * a load it first poses time 0, then the current time, which this reproduces from the base mesh.
 */
function posePolytube(mesh: SceneMeshDesc, time: number, preCycle: boolean): StaticVertex[] {
    return stepPolytube(mesh, stepPolytube(mesh, mesh.vertices, 0, preCycle), time, preCycle);
}

function stepPolytube(mesh: SceneMeshDesc, from: StaticVertex[], time: number, preCycle: boolean): StaticVertex[] {
    const { times, curvePoints, strandLast } = mesh.polytubeCache!;
    const [a, b, t] = curveInterpolation(times, time, preCycle);
    const [pa, pb] = [curvePoints[a]!, curvePoints[b]!];
    const center = (v: number) => new float3(pa[v * 3]! + (pb[v * 3]! - pa[v * 3]!) * t, pa[v * 3 + 1]! + (pb[v * 3 + 1]! - pa[v * 3 + 1]!) * t, pa[v * 3 + 2]! + (pb[v * 3 + 2]! - pa[v * 3 + 2]!) * t);
    const unit = (v: float3) => {
        const l = Math.hypot(v.x, v.y, v.z);
        return new float3(v.x / l, v.y / l, v.z / l);
    };
    return from.map((base, i) => {
        const cv = Math.floor(i / 4);
        const p = center(cv);
        const q = strandLast[cv] ? center(cv - 1) : center(cv + 1);
        const tangent = strandLast[cv] ? unit(new float3(p.x - q.x, p.y - q.y, p.z - q.z)) : unit(new float3(q.x - p.x, q.y - p.y, q.z - p.z));
        const normal = rotateVector(fromToRotation(unit(new float3(base.tangent.x, base.tangent.y, base.tangent.z)), tangent), base.normal);
        const r = base.curveRadius ?? 0;
        let position = new float3(p.x + r * normal.x, p.y + r * normal.y, p.z + r * normal.z);
        if (![position.x, position.y, position.z].every(Number.isFinite)) position = p;
        return { ...base, position, normal, tangent: new float4(tangent.x, tangent.y, tangent.z, 1) };
    });
}

/** from_to_rotation (Utils/Math/Quaternion.slang). */
function fromToRotation(v1: float3, v2: float3): quatf {
    const d = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    if (d < -0.999999) {
        let tmp = new float3(0, -v1.z, v1.y); // cross((1, 0, 0), v1)
        if (Math.hypot(tmp.x, tmp.y, tmp.z) < 0.000001) tmp = new float3(v1.z, 0, -v1.x); // cross((0, 1, 0), v1)
        const l = Math.hypot(tmp.x, tmp.y, tmp.z);
        return new quatf(tmp.x / l, tmp.y / l, tmp.z / l, 0); // rotate_angle_axis(pi, tmp)
    }
    if (d > 0.999999) return new quatf(0, 0, 0, 1);
    const c = new float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
    const l = Math.hypot(c.x, c.y, c.z, 1 + d);
    return new quatf(c.x / l, c.y / l, c.z / l, (1 + d) / l);
}

/** A cached curve's positions at `time` (calculateInterpolation, post-infinity Constant); radii and texcoords stay. */
function sampleCurveCache(curve: SceneCurveDesc, time: number, preCycle: boolean): Float32Array {
    const { times: ts, positions } = curve.vertexCache!;
    const [a, b, t] = curveInterpolation(ts, time, preCycle);
    const out = curve.positionsRadii.slice();
    const [pa, pb] = [positions[a]!, positions[b]!];
    for (let v = 0; v < out.length / 4; v++) for (let k = 0; k < 3; k++) out[v * 4 + k] = pa[v * 3 + k]! + (pb[v * 3 + k]! - pa[v * 3 + k]!) * t;
    return out;
}

/**
 * AnimatedVertexCache's mesh interpolation (calculateInterpolation + UpdateMeshVertices): looped
 * past the last sample, held or cycled before the first; positions and tangents lerp, normals and
 * tangent directions renormalize, texcoords stay the base mesh's.
 */
export function sampleVertexCache(cache: { times: number[]; frames: StaticVertex[][] }, time: number, preCycle: boolean, base: StaticVertex[], loop = true): StaticVertex[] {
    const ts = cache.times;
    let [a, b, t] = [0, 0, 0];
    if (Number.isFinite(time)) {
        time = Math.max(time, 0);
        const last = ts[ts.length - 1]!;
        // Post-infinity: Cycle while looping, else Constant (AnimatedVertexCache::updateMeshInterpolation).
        if (time > last) time = loop ? time % last : last;
        if (time <= ts[0]!) {
            if (preCycle) [a, b, t] = [ts.length - 1, 0, time / ts[0]!];
        } else {
            b = ts.findIndex((x) => x >= time);
            a = b - 1;
            t = (time - ts[a]!) / (ts[b]! - ts[a]!);
        }
    }
    const [fa, fb] = [cache.frames[a]!, cache.frames[b]!];
    const lerp3 = (x: float3, y: float3) => new float3(x.x + (y.x - x.x) * t, x.y + (y.y - x.y) * t, x.z + (y.z - x.z) * t);
    const unit = (v: float3) => {
        const l = Math.hypot(v.x, v.y, v.z) || 1;
        return new float3(v.x / l, v.y / l, v.z / l);
    };
    return base.map((v, i) => {
        const [va, vb] = [fa[i]!, fb[i]!];
        const tan = unit(lerp3(new float3(va.tangent.x, va.tangent.y, va.tangent.z), new float3(vb.tangent.x, vb.tangent.y, vb.tangent.z)));
        return {
            position: lerp3(va.position, vb.position),
            normal: unit(lerp3(va.normal, vb.normal)),
            tangent: new float4(tan.x, tan.y, tan.z, va.tangent.w + (vb.tangent.w - va.tangent.w) * t),
            texCrd: v.texCrd,
        };
    });
}

/** Mirrors Scene::Metadata: optional settings an importer found in the asset. */
export interface SceneMetadata {
    fNumber?: number;
    filmISO?: number;
    shutterSpeed?: number;
    samplesPerPixel?: number;
    maxDiffuseBounces?: number;
    maxSpecularBounces?: number;
    maxTransmissionBounces?: number;
    maxVolumeBounces?: number;
}

export interface SceneMeshDesc {
    vertices: StaticVertex[];
    indices: Uint32Array;
    materialID: number;
    /** World transform (row-major float4x4); identity if omitted. */
    transform?: float4x4;
    /** Animated scenes: index into the node graph whose global matrix drives this
     *  mesh's world transform (overrides `transform` each animated frame). */
    nodeID?: number;
    /** Skinned meshes: per-vertex joint binding; skinned to world in animate(). */
    skin?: SkinDesc;
    /** Morph-target meshes: blend-shape deltas applied before skinning in animate(). */
    morph?: MorphDesc;
    /** Vertex cache (AnimatedVertexCache's CachedMesh): per-sample vertices, times in seconds. */
    vertexCache?: { times: number[]; frames: StaticVertex[][] };
    /** Poly-tube curve cache (CachedCurve, PolyTube): per-sample ring centers, 4 tube vertices per center. */
    polytubeCache?: { times: number[]; curvePoints: Float32Array[]; strandLast: Uint8Array };
    /** Tangents SceneBuilder.resolve still has to generate (MikkTSpace + vertex merge); unset keeps them. */
    tangentSpace?: import("./TangentSpace.js").TangentSpaceMode;
}

/** Tessellated curve geometry (linear swept spheres; CurveTessellation). */
export interface SceneCurveDesc {
    /** xyz position + radius per vertex, concatenated (4 floats/vertex). */
    positionsRadii: Float32Array;
    /** uv per vertex (2 floats/vertex); zeros if absent. */
    texCrds: Float32Array | null;
    /** Segment-start indices (local to this curve's vertices). */
    indices: Uint32Array;
    materialID: number;
    /** World transform; identity if omitted. */
    transform?: float4x4;
    /** Curve vertex cache (CachedCurve): per-sample positions (xyz per vertex), times in seconds. */
    vertexCache?: { times: number[]; positions: Float32Array[] };
}

/** Mirrors MaterialSystem::MaterialStats. */
export interface MaterialStats {
    materialTypeCount: number;
    materialCount: number;
    materialOpaqueCount: number;
    materialMemoryInBytes: number;
    textureCount: number;
    textureCompressedCount: number;
    textureTexelCount: number;
    textureTexelChannelCount: number;
    textureMemoryInBytes: number;
}

/** Mirrors Scene::SceneStats (see Scene.getSceneStats). */
export interface SceneStats {
    meshCount: number;
    meshInstanceCount: number;
    meshInstanceOpaqueCount: number;
    transformCount: number;
    uniqueTriangleCount: number;
    uniqueVertexCount: number;
    instancedTriangleCount: number;
    instancedVertexCount: number;
    indexMemoryInBytes: number;
    vertexMemoryInBytes: number;
    geometryMemoryInBytes: number;
    animationMemoryInBytes: number;
    curveCount: number;
    curveInstanceCount: number;
    uniqueCurveSegmentCount: number;
    uniqueCurvePointCount: number;
    instancedCurveSegmentCount: number;
    instancedCurvePointCount: number;
    curveIndexMemoryInBytes: number;
    curveVertexMemoryInBytes: number;
    sdfGridCount: number;
    sdfGridDescriptorCount: number;
    sdfGridInstancesCount: number;
    sdfGridMemoryInBytes: number;
    customPrimitiveCount: number;
    materials: MaterialStats;
    blasGroupCount: number;
    blasCount: number;
    blasCompactedCount: number;
    blasOpaqueCount: number;
    blasGeometryCount: number;
    blasOpaqueGeometryCount: number;
    blasMemoryInBytes: number;
    blasScratchMemoryInBytes: number;
    tlasCount: number;
    tlasMemoryInBytes: number;
    tlasScratchMemoryInBytes: number;
    activeLightCount: number;
    totalLightCount: number;
    pointLightCount: number;
    directionalLightCount: number;
    rectLightCount: number;
    discLightCount: number;
    sphereLightCount: number;
    distantLightCount: number;
    lightsMemoryInBytes: number;
    envMapMemoryInBytes: number;
    emissiveMemoryInBytes: number;
    gridVolumeCount: number;
    gridVolumeMemoryInBytes: number;
    gridCount: number;
    gridVoxelCount: number;
    gridMemoryInBytes: number;
}

/** Mirrors LightCollection::MeshLightStats. */
export interface MeshLightStats {
    meshLightCount: number;
    triangleCount: number;
    meshesTextured: number;
    trianglesTextured: number;
    trianglesCulled: number;
    trianglesActiveUniform: number;
    trianglesActiveTextured: number;
    trianglesActive: number;
}

export interface SceneMaterialDesc {
    /** Material name; used by Scene.getMaterial(name). */
    name?: string;
    header?: Partial<MaterialHeaderDesc>;
    /** Parameters of a basic (standard/cloth/hair/PBRT) material. */
    basic: BasicMaterialDesc;
    /** Measured MERL BRDF; its table and albedo LUT live in the shared material buffer. */
    merl?: MERLBRDF;
    /** Measured RGL BSDF; its tables, CDFs and albedo LUT live in the shared material buffer. */
    rgl?: RGLMeasurement;
    /** Several MERL BRDFs selected per texel by an index map (MERLMixMaterial). */
    merlMix?: MERLMixData;
}

/**
 * Mirrors StandardMaterial::updateDeltaSpecularFlag: a material is delta
 * specular when it has no rough lobe and nothing diffuse (or is fully
 * transmissive). Only StandardMaterial computes it; the flag feeds the path
 * tracer's coherence hints.
 */
function isDeltaSpecularStandard(header: MaterialHeaderDesc, basic: BasicMaterialDesc): boolean {
    if ((header.materialType ?? MaterialType.Standard) !== MaterialType.Standard) return false;
    const hasTexture = (handle: number | undefined) => handle !== undefined && ((handle >>> 29) & 0x3) === TextureHandleMode.Texture;
    const base = basic.baseColor;
    let isNonDiffuse = !hasTexture(basic.texBaseColor) && (base ? base.x === 0 && base.y === 0 && base.z === 0 : false) && (basic.diffuseTransmission ?? 0) === 0;
    const isFullyTransmissive = (basic.specularTransmission ?? 0) >= 1;
    let isDelta = false;
    if ((basic.shadingModel ?? ShadingModel.MetalRough) === ShadingModel.MetalRough && !hasTexture(basic.texSpecular)) {
        // Specular green is roughness and blue is metallic in metal-rough mode.
        const spec = basic.specular ?? new float4(0, 0.5, 0, 0);
        isDelta = f16Round(spec.y) === 0;
        if (f16Round(spec.z) >= 1) isNonDiffuse = true;
    }
    return isDelta && (isNonDiffuse || isFullyTransmissive);
}

/** The value a parameter has after the float16 packing native compares. */
function f16Round(v: number): number {
    return float16ToFloat32(float32ToFloat16(v));
}

export class Scene {
    /** Mirrors Scene's camera list; `camera` is the selected one (Scene::getCamera/selectCamera). */
    private cameraList: Camera[] = [new Camera()];
    private activeCameraIndex = 0;
    /** Index of the camera bound to cameraNodeID (the imported, animated one). */
    private animatedCameraIndex = 0;

    get camera(): Camera {
        return this.cameraList[this.activeCameraIndex]!;
    }

    /** Mirrors the Python `scene.camera = cam` property: selects `cam` from the camera list. */
    set camera(camera: Camera) {
        this.selectCamera(camera);
    }

    /** Mirrors Scene::getCameras. */
    getCameras(): Camera[] {
        return [...this.cameraList];
    }

    /** Python-facing alias (`scene.cameras`). */
    get cameras(): Camera[] {
        return this.getCameras();
    }

    /** Mirrors Scene::selectCamera (by index or instance); the new camera keeps the frame's aspect ratio. */
    selectCamera(camera: Camera | number): void {
        const index = typeof camera === "number" ? camera : this.cameraList.indexOf(camera);
        if (index < 0 || index >= this.cameraList.length) throw new RuntimeError(`Scene.selectCamera: camera ${typeof camera === "number" ? camera : "instance"} is not in the scene`);
        if (index === this.activeCameraIndex) return;
        const aspect = this.camera.getAspectRatio();
        this.activeCameraIndex = index;
        this.camera.setAspectRatio(aspect);
    }

    /** Index of the selected camera in getCameras() (Scene::mSelectedCamera). */
    getSelectedCameraIndex(): number {
        return this.activeCameraIndex;
    }

    /** Saved camera poses (Scene::Viewpoint); index 0 is the default one. */
    private readonly viewpoints: { index: number; position: float3; target: float3; up: float3 }[] = [];
    private currentViewpoint = 0;

    /** Mirrors Scene::addViewpoint: the current camera pose, or the given one, becomes the current viewpoint. */
    addViewpoint(position?: float3, target?: float3, up?: float3, cameraIndex = this.activeCameraIndex): void {
        const camera = this.camera;
        const v = (p: { x: number; y: number; z: number }) => new float3(Number(p.x), Number(p.y), Number(p.z));
        this.viewpoints.push({ index: cameraIndex, position: v(position ?? camera.getPosition()), target: v(target ?? camera.getTarget()), up: v(up ?? camera.getUpVector()) });
        this.currentViewpoint = this.viewpoints.length - 1;
    }
    /** Mirrors Scene::removeViewpoint (the default viewpoint stays). */
    removeViewpoint(): void {
        if (this.currentViewpoint === 0) {
            Logger.warning("Cannot remove default viewpoint.");
            return;
        }
        this.viewpoints.splice(this.currentViewpoint, 1);
        this.currentViewpoint = Math.min(this.currentViewpoint, this.viewpoints.length - 1);
    }
    /** Mirrors Scene::selectViewpoint: selects its camera and moves it to the saved pose. */
    selectViewpoint(index: number): void {
        const vp = this.viewpoints[index];
        if (!vp) {
            Logger.warning("Viewpoint does not exist.");
            return;
        }
        this.selectCamera(vp.index);
        this.camera.setPosition(vp.position);
        this.camera.setTarget(vp.target);
        this.camera.setUpVector(vp.up);
        this.currentViewpoint = index;
    }
    getViewpointCount(): number {
        return this.viewpoints.length;
    }
    getCurrentViewpoint(): number {
        return this.currentViewpoint;
    }
    /** The "Save Viewpoints" text: one camera keyframe per viewpoint over `animationLength`, closing on the first. */
    getViewpointsScript(animationLength: number): string {
        const f = (v: float3) => `float3(${v.x}, ${v.y}, ${v.z})`;
        const line = (t: number, vp: (typeof this.viewpoints)[number]) => `${t}, Transform(position = ${f(vp.position)}, target = ${f(vp.target)}, up = ${f(vp.up)})`;
        const lines = this.viewpoints.map((vp, i) => line((animationLength * i) / this.viewpoints.length, vp));
        lines.push(line(animationLength, this.viewpoints[0]!));
        return lines.join("\n") + "\n";
    }

    getAnimatedCameraIndex(): number {
        return this.animatedCameraIndex;
    }

    /** SceneBuilder hookup: every camera the scene defines, the selected one, and which one the node animates. */
    setCameraList(cameras: Camera[], active: number, animated = 0): void {
        if (cameras.length === 0) return;
        this.cameraList = cameras;
        this.activeCameraIndex = Math.min(Math.max(active, 0), cameras.length - 1);
        this.animatedCameraIndex = animated;
        // Native adds the default viewpoint (the selected camera's pose) at scene creation.
        this.viewpoints.length = 0;
        this.addViewpoint();
        if (this.cameraNodeID !== undefined && cameras[animated]) cameras[animated]!.hasAnimation = true;
        // Native's first scene update poses animated cameras and lights at time 0.
        if (this.animData && this.hasAnimatedCameraOrLights && this.animationEnabled) this.updateAnimatedCameraAndLights(evaluateGlobals(this.animData, 0));
    }
    readonly gridVolumes: import("./Volume/GridVolume.js").GridVolume[] = [];

    /**
     * Python `scene.stats`: native's flat SceneStats dict (materials fields inlined), plus the
     * web's own short counters (instances, materials, textures, vertices, triangles).
     */
    get stats(): { instances: number; materials: number; textures: number; vertices: number; triangles: number } & Omit<SceneStats, "materials"> & MaterialStats {
        const { materials, ...rest } = this.getSceneStats();
        return {
            ...rest,
            ...materials,
            instances: this.instanceCount,
            materials: this.materialCount,
            textures: this.textureCount,
            vertices: this.vertexTotal,
            triangles: this.triangleTotal,
        };
    }

    /**
     * Mirrors Scene::getSceneStats (native field names). §9: the web has no BLAS/TLAS; its software
     * BVH is reported as one TLAS-less BLAS group, and memory figures are packed resource sizes.
     */
    getSceneStats(): SceneStats {
        const size = (...names: string[]) => names.reduce((sum, n) => sum + (this.buffers[n]?.size ?? 0), 0);
        const meshes = this.lcMeshes;
        const firstOfMesh = new Map<number, number>();
        this.meshIDs.forEach((id, i) => firstOfMesh.has(id) || firstOfMesh.set(id, i));
        const opaque = this.materialDescs.map((m) => {
            const header: MaterialHeaderDesc = { materialType: MaterialType.Standard, ...m.header };
            return (header.alphaMode ?? this.deriveAlphaMode(header, m.basic)) === AlphaMode.Opaque;
        });
        const lightsOf = (t: LightType) => this.analyticLights.filter((l) => l.type === t).length;
        // Textures as loaded (TextureManager sources); memory is what the bucket arrays take.
        const tm = this.lcTextureManager;
        let texelCount = 0, channelCount = 0, compressed = 0;
        for (let id = 0; id < tm.count; id++) {
            const src = tm.getSource(id)!;
            const c = src.compressed;
            const levels = c ? c.levels.length : src.mips ? src.mips.length + 1 : Math.floor(Math.log2(Math.max(src.bitmap.width, src.bitmap.height))) + 1;
            const [w, h] = c ? [c.width, c.height] : [src.bitmap.width, src.bitmap.height];
            let texels = 0;
            for (let mip = 0; mip < levels; mip++) texels += Math.max(1, w >> mip) * Math.max(1, h >> mip);
            texelCount += texels;
            channelCount += texels * (c ? getFormatChannelCount(c.format) : 4);
            if (c) compressed++;
        }
        const grids = [...new Set(this.gridVolumes.flatMap((v) => [...v.getGridSequence("density"), ...v.getGridSequence("emission")]))];
        const curveSegments = this.sceneCurves.reduce((n, c) => n + c.indices.length, 0);
        const curvePoints = this.sceneCurves.reduce((n, c) => n + c.positionsRadii.length / 4, 0);
        const bvhMemory = size("bvhNodes");
        return {
            meshCount: firstOfMesh.size,
            meshInstanceCount: meshes.length,
            meshInstanceOpaqueCount: meshes.filter((m) => opaque[m.materialID] ?? true).length,
            transformCount: meshes.length + this.sdfGrids.length + this.sceneCurves.length,
            uniqueTriangleCount: [...firstOfMesh.values()].reduce((n, i) => n + meshes[i]!.indices.length / 3, 0),
            uniqueVertexCount: [...firstOfMesh.values()].reduce((n, i) => n + meshes[i]!.vertices.length, 0),
            instancedTriangleCount: this.triangleTotal,
            instancedVertexCount: this.vertexTotal,
            indexMemoryInBytes: size("indices"),
            vertexMemoryInBytes: size("vertices", "prevVertices"),
            geometryMemoryInBytes: size("meshes", "geometryInstances", "drawIDs"),
            animationMemoryInBytes: size("worldMatrices", "prevWorldMatrices"),
            curveCount: this.sceneCurves.length,
            curveInstanceCount: this.sceneCurves.length,
            uniqueCurveSegmentCount: curveSegments,
            uniqueCurvePointCount: curvePoints,
            instancedCurveSegmentCount: curveSegments,
            instancedCurvePointCount: curvePoints,
            curveIndexMemoryInBytes: size("curveIndices"),
            curveVertexMemoryInBytes: size("curveVertices", "curves"),
            sdfGridCount: new Set(this.sdfGrids.map((d) => d.grid)).size,
            sdfGridDescriptorCount: new Set(this.sdfGrids.map((d) => d.grid)).size,
            sdfGridInstancesCount: this.sdfGrids.length,
            sdfGridMemoryInBytes: Object.keys(this.buffers).filter((n) => n.startsWith("sdf")).reduce((n, k) => n + this.buffers[k]!.size, 0) + (this.sdfAtlasTexture?.getTextureSizeInBytes() ?? 0),
            customPrimitiveCount: this.customPrimitives.length,
            materials: {
                materialTypeCount: this.materialTypes.size,
                materialCount: this.materialDescs.length,
                materialOpaqueCount: opaque.filter(Boolean).length,
                materialMemoryInBytes: size("materialData"),
                textureCount: tm.count,
                textureCompressedCount: compressed,
                textureTexelCount: texelCount,
                textureTexelChannelCount: channelCount,
                textureMemoryInBytes: this.builtTextureCount > 0 ? this.textureBuckets.reduce((n, t) => n + t.getTextureSizeInBytes(), 0) : 0,
            },
            blasGroupCount: bvhMemory > 0 ? 1 : 0,
            blasCount: bvhMemory > 0 ? 1 : 0,
            blasCompactedCount: 0,
            blasOpaqueCount: bvhMemory > 0 && meshes.every((m) => opaque[m.materialID] ?? true) ? 1 : 0,
            blasGeometryCount: meshes.length,
            blasOpaqueGeometryCount: meshes.filter((m) => opaque[m.materialID] ?? true).length,
            blasMemoryInBytes: bvhMemory,
            blasScratchMemoryInBytes: 0,
            tlasCount: 0,
            tlasMemoryInBytes: 0,
            tlasScratchMemoryInBytes: 0,
            activeLightCount: this.activeLights.length,
            totalLightCount: this.analyticLights.length,
            pointLightCount: lightsOf(LightType.Point),
            directionalLightCount: lightsOf(LightType.Directional),
            rectLightCount: lightsOf(LightType.Rect),
            discLightCount: lightsOf(LightType.Disc),
            sphereLightCount: lightsOf(LightType.Sphere),
            distantLightCount: lightsOf(LightType.Distant),
            lightsMemoryInBytes: size("lights"),
            envMapMemoryInBytes: this.envMap?.texture.getTextureSizeInBytes() ?? 0,
            emissiveMemoryInBytes: size("emissiveTriangles", "emissiveFlux", "emissiveActiveTriangles", "emissiveTriToActive", "emissiveMeshData", "emissivePerMeshInstanceOffset"),
            gridVolumeCount: this.gridVolumes.length,
            gridVolumeMemoryInBytes: this.gridVolumes.length > 0 ? size("gridVolumesData") : 0,
            gridCount: grids.length,
            gridVoxelCount: grids.reduce((n, g) => n + g.voxelCount, 0),
            gridMemoryInBytes: grids.reduce((n, g) => n + g.gridBuffer.byteLength, 0),
        };
    }

    /** Mirrors LightCollection::getStats (MeshLightStats). */
    getMeshLightStats(): MeshLightStats {
        const stats: MeshLightStats = { meshLightCount: this.emissiveMeshCount, triangleCount: this.emissiveTriangleCount, meshesTextured: 0, trianglesTextured: 0, trianglesCulled: 0, trianglesActiveUniform: 0, trianglesActiveTextured: 0, trianglesActive: 0 };
        const textured = (materialID: number) => this.materialDescs[materialID]?.basic.texEmissive !== undefined;
        for (let i = 0; i < this.emissiveMeshCount; i++) {
            const [triOffset, triCount, materialID] = [this.emissiveMeshData[i * 4 + 1]!, this.emissiveMeshData[i * 4 + 2]!, this.emissiveMeshData[i * 4 + 3]!];
            if (textured(materialID)) {
                stats.meshesTextured++;
                stats.trianglesTextured += triCount;
            }
            for (let t = triOffset; t < triOffset + triCount; t++) {
                if (this.emissiveFluxes[t] === 0) stats.trianglesCulled++;
                else if (textured(materialID)) stats.trianglesActiveTextured++;
                else stats.trianglesActiveUniform++;
            }
        }
        stats.trianglesActive = stats.trianglesActiveUniform + stats.trianglesActiveTextured;
        return stats;
    }

    /** The "Statistics" text of Scene::renderUI. */
    getSceneStatsText(): string {
        const s = this.getSceneStats();
        const m = s.materials;
        const b = formatByteSize;
        const bounds = this.worldBounds;
        const total =
            s.indexMemoryInBytes + s.vertexMemoryInBytes + s.geometryMemoryInBytes + s.animationMemoryInBytes + s.curveIndexMemoryInBytes + s.curveVertexMemoryInBytes + s.sdfGridMemoryInBytes +
            m.materialMemoryInBytes + m.textureMemoryInBytes + s.blasMemoryInBytes + s.blasScratchMemoryInBytes + s.tlasMemoryInBytes + s.tlasScratchMemoryInBytes +
            s.lightsMemoryInBytes + s.envMapMemoryInBytes + s.emissiveMemoryInBytes + s.gridVolumeMemoryInBytes + s.gridMemoryInBytes;
        const lines = [
            `Path: ${this.importPaths.at(-1) ?? ""}`,
            `Bounds: (${bounds?.min.join(",") ?? "0,0,0"})-(${bounds?.max.join(",") ?? "0,0,0"})`,
            `Total scene memory: ${b(total)}`,
            "Geometry stats:",
            `  Mesh count: ${s.meshCount}`,
            `  Mesh instance count (total): ${s.meshInstanceCount}`,
            `  Mesh instance count (opaque): ${s.meshInstanceOpaqueCount}`,
            `  Mesh instance count (non-opaque): ${s.meshInstanceCount - s.meshInstanceOpaqueCount}`,
            `  Transform matrix count: ${s.transformCount}`,
            `  Unique triangle count: ${s.uniqueTriangleCount}`,
            `  Unique vertex count: ${s.uniqueVertexCount}`,
            `  Instanced triangle count: ${s.instancedTriangleCount}`,
            `  Instanced vertex count: ${s.instancedVertexCount}`,
            `  Index  buffer memory: ${b(s.indexMemoryInBytes)}`,
            `  Vertex buffer memory: ${b(s.vertexMemoryInBytes)}`,
            `  Geometry data memory: ${b(s.geometryMemoryInBytes)}`,
            `  Animation data memory: ${b(s.animationMemoryInBytes)}`,
            `  Curve count: ${s.curveCount}`,
            `  Curve instance count: ${s.curveInstanceCount}`,
            `  Unique curve segment count: ${s.uniqueCurveSegmentCount}`,
            `  Unique curve point count: ${s.uniqueCurvePointCount}`,
            `  Instanced curve segment count: ${s.instancedCurveSegmentCount}`,
            `  Instanced curve point count: ${s.instancedCurvePointCount}`,
            `  Curve index buffer memory: ${b(s.curveIndexMemoryInBytes)}`,
            `  Curve vertex buffer memory: ${b(s.curveVertexMemoryInBytes)}`,
            `  SDF grid count: ${s.sdfGridCount}`,
            `  SDF grid descriptor count: ${s.sdfGridDescriptorCount}`,
            `  SDF grid instances count: ${s.sdfGridInstancesCount}`,
            `  SDF grid memory: ${b(s.sdfGridMemoryInBytes)}`,
            `  Custom primitive count: ${s.customPrimitiveCount}`,
            "",
            "Raytracing stats (software BVH):",
            `  BLAS groups: ${s.blasGroupCount}`,
            `  BLAS count (total): ${s.blasCount}`,
            `  BLAS count (compacted): ${s.blasCompactedCount}`,
            `  BLAS count (opaque): ${s.blasOpaqueCount}`,
            `  BLAS count (non-opaque): ${s.blasCount - s.blasOpaqueCount}`,
            `  BLAS geometries (total): ${s.blasGeometryCount}`,
            `  BLAS geometries (opaque): ${s.blasOpaqueGeometryCount}`,
            `  BLAS geometries (non-opaque): ${s.blasGeometryCount - s.blasOpaqueGeometryCount}`,
            `  BLAS memory (final): ${b(s.blasMemoryInBytes)}`,
            `  BLAS memory (scratch): ${b(s.blasScratchMemoryInBytes)}`,
            `  TLAS count: ${s.tlasCount}`,
            `  TLAS memory (final): ${b(s.tlasMemoryInBytes)}`,
            `  TLAS memory (scratch): ${b(s.tlasScratchMemoryInBytes)}`,
            "",
            "Materials stats:",
            `  Material types: ${m.materialTypeCount}`,
            `  Material count (total): ${m.materialCount}`,
            `  Material count (opaque): ${m.materialOpaqueCount}`,
            `  Material count (non-opaque): ${m.materialCount - m.materialOpaqueCount}`,
            `  Material memory: ${b(m.materialMemoryInBytes)}`,
            `  Texture count (total): ${m.textureCount}`,
            `  Texture count (compressed): ${m.textureCompressedCount}`,
            `  Texture texel count: ${m.textureTexelCount}`,
            `  Texture memory: ${b(m.textureMemoryInBytes)}`,
            `  Bytes/texel (average): ${(m.textureTexelCount > 0 ? m.textureMemoryInBytes / m.textureTexelCount : 0).toFixed(2)}`,
            // Native divides by zero here for texture-less scenes (prints nan).
            `  Channels/texel (average): ${(m.textureTexelChannelCount / m.textureTexelCount).toFixed(2)}`,
            "",
            "Analytic light stats:",
            `  Active light count: ${s.activeLightCount}`,
            `  Total light count: ${s.totalLightCount}`,
            `  Point light count: ${s.pointLightCount}`,
            `  Directional light count: ${s.directionalLightCount}`,
            `  Rect light count: ${s.rectLightCount}`,
            `  Disc light count: ${s.discLightCount}`,
            `  Sphere light count: ${s.sphereLightCount}`,
            `  Distant light count: ${s.distantLightCount}`,
            `  Analytic lights memory: ${b(s.lightsMemoryInBytes)}`,
            "",
            "Emissive light stats:",
        ];
        if (this.emissiveMeshCount > 0) {
            const e = this.getMeshLightStats();
            lines.push(
                `  Active triangle count: ${e.trianglesActive}`,
                `  Active uniform triangle count: ${e.trianglesActiveUniform}`,
                `  Active textured triangle count: ${e.trianglesActiveTextured}`,
                "  Details:",
                `    Total mesh count: ${e.meshLightCount}`,
                `    Textured mesh count: ${e.meshesTextured}`,
                `    Total triangle count: ${e.triangleCount}`,
                `    Texture triangle count: ${e.trianglesTextured}`,
                `    Culled triangle count: ${e.trianglesCulled}`,
                `  Emissive lights memory: ${b(s.emissiveMemoryInBytes)}`,
            );
        } else lines.push("  N/A");
        lines.push("", "Environment map:");
        if (this.envMap) lines.push(`  Filename: ${this.envMap.path}`, `  Resolution: ${this.envMap.texture.width}x${this.envMap.texture.height}`, `  Texture memory: ${b(s.envMapMemoryInBytes)}`);
        else lines.push("  N/A");
        lines.push(
            "",
            "Grid volume stats:",
            `  Grid volume count: ${s.gridVolumeCount}`,
            `  Grid volume memory: ${b(s.gridVolumeMemoryInBytes)}`,
            "",
            "Grid stats:",
            `  Grid count: ${s.gridCount}`,
            `  Grid voxel count: ${s.gridVoxelCount}`,
            `  Grid memory: ${b(s.gridMemoryInBytes)}`,
            "",
        );
        return lines.join("\n");
    }

    /** World-space geometry AABB (BVH root); null for geometry-less scenes. */
    worldBounds: { min: [number, number, number]; max: [number, number, number] } | null = null;

    private gridCount = 0;
    private grid0Stats: { minIndex: [number, number, number]; minValue: number; maxIndex: [number, number, number]; maxValue: number } | null = null;

    /** Stats of grid 0 (diagnostics/tests). */
    get gridStats(): { minIndex: [number, number, number]; minValue: number; maxIndex: [number, number, number]; maxValue: number } | null {
        return this.grid0Stats;
    }
    private buffers: Record<string, Buffer> = {};
    /** Material texture arrays (TextureManager buckets); the shader binds kMaxTextureBuckets. */
    private textureBuckets: Texture[] = [];
    private texInfoTexture!: Texture;
    private dummyTexture: Texture;
    /** IES profile shared by materials with `lightProfileEnabled` (MaterialSystem::mpLightProfile). */
    lightProfile: LightProfile | null = null;
    private texture3D: Texture;
    private sampler: Sampler;
    private materialCount = 0;
    /** Mirrors Scene::getMaterialCount. */
    getMaterialCount(): number {
        return this.materialCount;
    }
    private instanceCount = 0;
    /** Native mesh ID per triangle instance (instances of one builder mesh share it). */
    private meshIDs = new Uint32Array(0);
    /** Per web mesh (instance): material ID and counts, for getGeometryIDsForMaterial / get_mesh. */
    private meshMaterialIDs: number[] = [];
    private meshCounts: { vertexCount: number; indexCount: number }[] = [];
    /** Mesh vertices/triangles uploaded (Scene::getSceneStats meshVertexCount / meshTriangleCount). */
    private vertexTotal = 0;
    private triangleTotal = 0;
    private maxPrimitiveCount = 0;
    private textureCount = 1;
    /** Textures in the built buckets (textureCount is clamped to 1). */
    private builtTextureCount = 0;
    private drawList: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[] = [];

    private lightCount = 0;
    /** Analytic light descriptors (RTXDI needs types/order). */
    readonly analyticLights: AnalyticLight[] = [];
    /** Mirrors Scene::getImportPaths: the scene file, then nested imports. */
    readonly importPaths: string[] = [];
    emissiveActiveTriangleCount = 0;
    private bvhTrisOffset = 0;
    /** Last animated-frame BVH, refit on the next animation step. */
    private animatedBvh: import("./SoftwareRT/Bvh.js").BvhBuildResult | null = null;
    private invTransposeOffset = 0;
    /** Geometry instance (and node) index of SDF grid 0; mesh instances come first. */
    private sdfInstanceBase = 0;
    /** Mirrors Scene::setCameraControlsEnabled (the viewer's camera controller checks it). */
    cameraControlsEnabled = true;
    // Animation state (retained only for animated scenes; null otherwise).
    private sourceMeshes: SceneMeshDesc[] | null = null;
    private animData: SceneAnimations | null = null;
    private animNodeCount = 0;
    // Prev-frame motion state: matrices/deformed verts as uploaded last frame.
    private lastWorldMats: Float32Array | null = null;
    private pendingPrevVerts = new Map<number, Float32Array>();
    /** Node whose animated global drives the camera (glTF camera); undefined if none. */
    private cameraNodeID?: number;
    /** True once any analytic light or the camera is bound to an animated node. */
    private hasAnimatedCameraOrLights = false;
    private bvhCapacityBytes = 0;
    readonly sdfGrids: SceneSDFGridDesc[];
    private sdfInstanceFirst = 0;
    private sdfAtlasTexture: Texture | null = null;
    private sdfSampler: Sampler | null = null;
    private sbsResources: { aabbs: Buffer; indirection: Texture; bricks: Texture; sampler: Sampler } | null = null;
    private svsResources: { voxels: Buffer } | null = null;
    private svoResources: { svo: Buffer } | null = null;
    private sdfBvhBuffers: { buf: Buffer; primOffset: number } | null = null;
    /** Distinct SBS grids packed into the one sdfGrid0 binding (built with the SBS resources). */
    private sbsPacked: { grids: SDFSBS[]; packed: PackedSBS } | null = null;
    private displacementTexture: Texture | null = null;
    private curveInstanceFirst = 0;
    private curveDescs: SceneCurveDesc[] = [];
    private curveBvhOffset = 0;
    private curvePrimOffset = 0;
    private curveBvhBytes: Float32Array | null = null;
    private sceneCurves: SceneCurveDesc[] = [];
    private displacedBvhOffset = 0;
    private displacedPrimOffset = 0;
    private hasDisplaced = false;
    private envMap: EnvMap | null = null;
    private hasEmissiveMaterials = false;
    private materialTypes = new Set<MaterialType>();
    private materialDescs: SceneMaterialDesc[] = [];
    /** Byte offsets of each MERL material's table and albedo LUT in materialBuffer0. */
    private readonly merlOffsets = new Map<number, { data: number; lut: number }>();
    /** Element offsets of each RGL material's tables in materialBuffer0. */
    private readonly rglOffsets = new Map<number, Record<string, number>>();
    /** Byte offsets of each MERLMix material's regions in materialBuffer0. */
    private readonly merlMixOffsets = new Map<number, { data: number; stride: number; extra: number; indexMap: number; lut: number }>();
    private lcMeshes: SceneMeshDesc[] = [];
    private lcTextureManager: TextureManager = new TextureManager();
    private emissiveTriangleCount = 0;
    private emissiveMeshCount = 0;
    /** MeshLightData per mesh light: instance, triangle offset, triangle count, material. */
    private emissiveMeshData: Uint32Array = new Uint32Array(0);
    /** Bumped whenever the LightCollection buffers are rebuilt (native ILightCollection::UpdateFlags). */
    emissiveVersion = 0;
    /** Emissive-mesh world matrices of the last animate() (change detection, mirrors isMatrixChanged). */
    private lastEmissiveMats = new Map<number, Float32Array>();
    private emissiveFluxes = new Float32Array(0);
    private emissiveTriangles: EmissiveTriangleInput[] = [];

    /** Vertices normalize to f32 up front (native holds f32 StaticVertexData), in place, once per vertex array. */
    private static roundVertices(meshes: SceneMeshDesc[]): void {
        const fr = Math.fround;
        for (const mesh of meshes) {
            if (roundedVertexArrays.has(mesh.vertices)) continue;
            roundedVertexArrays.add(mesh.vertices);
            for (const v of mesh.vertices) {
                const { position: p, normal: n, tangent: t, texCrd: uv } = v;
                p.x = fr(p.x); p.y = fr(p.y); p.z = fr(p.z);
                n.x = fr(n.x); n.y = fr(n.y); n.z = fr(n.z);
                t.x = fr(t.x); t.y = fr(t.y); t.z = fr(t.z); t.w = fr(t.w);
                uv.x = fr(uv.x); uv.y = fr(uv.y);
                if (v.curveRadius !== undefined) v.curveRadius = fr(v.curveRadius);
            }
        }
    }

    /** The BVH's world-space triangles; displaced meshes are excluded (they intersect via their own AABB region). */
    static collectBvhGeometry(meshes: SceneMeshDesc[], materials: SceneMaterialDesc[]) {
        const bvhTris: BvhTriangle[] = [];
        const displacedAabbs: { min: [number, number, number]; max: [number, number, number] }[] = [];
        const displacedEntries: number[] = [];
        meshes.forEach((mesh, meshID) => {
            const m = mesh.transform ?? float4x4.identity();
            const mat = materials[mesh.materialID];
            const displaced = mat?.basic.texDisplacement !== undefined;
            // Conservative displacement range along the normal: mapValue([0,1]).
            const scaleD = mat?.basic.displacementScale ?? 0;
            const biasD = mat?.basic.displacementOffset ?? 0;
            const margin = Math.max(Math.abs(biasD), Math.abs(scaleD + biasD)) + 1e-3;
            for (let p = 0; p < mesh.indices.length / 3; p++) {
                const v0 = transformPoint(m, mesh.vertices[mesh.indices[p * 3]!]!.position);
                const v1 = transformPoint(m, mesh.vertices[mesh.indices[p * 3 + 1]!]!.position);
                const v2 = transformPoint(m, mesh.vertices[mesh.indices[p * 3 + 2]!]!.position);
                if (displaced) {
                    displacedAabbs.push({
                        min: [Math.min(v0.x, v1.x, v2.x) - margin, Math.min(v0.y, v1.y, v2.y) - margin, Math.min(v0.z, v1.z, v2.z) - margin],
                        max: [Math.max(v0.x, v1.x, v2.x) + margin, Math.max(v0.y, v1.y, v2.y) + margin, Math.max(v0.z, v1.z, v2.z) + margin],
                    });
                    displacedEntries.push(((meshID & 0xff) << 24) | p);
                } else {
                    bvhTris.push({ v0, v1, v2, instanceIndex: meshID, primitiveIndex: p });
                }
            }
        });
        return { bvhTris, displacedAabbs, displacedEntries };
    }

    /**
     * `new Scene(...)` with the triangle BVH built on the worker pool for large scenes (native
     * builds the scene with TaskManager); the tree is byte-identical to the serial build.
     */
    static async create(...args: ConstructorParameters<typeof Scene>): Promise<Scene> {
        const [, meshes, materials = []] = args;
        Scene.roundVertices(meshes);
        const geometry = Scene.collectBvhGeometry(meshes, materials);
        const pool = WorkerPool.get();
        const bvh =
            geometry.bvhTris.length >= 100_000 && pool.threadCount > 1
                ? await buildBvhParallel(geometry.bvhTris, (input) => pool.run("buildBvhSubtree", input, [input.bmin.buffer, input.bmax.buffer, input.cent.buffer]))
                : undefined;
        const withBvh = [...args] as ConstructorParameters<typeof Scene>;
        withBvh[11] = { geometry, bvh };
        return new Scene(...withBvh);
    }

    constructor(
        public readonly device: Device,
        meshes: SceneMeshDesc[],
        materials: SceneMaterialDesc[],
        lights: AnalyticLight[] = [],
        textureManager: TextureManager = new TextureManager(),
        sdfGrids: SceneSDFGridDesc[] = [],
        nodes: SceneNode[] = [],
        animations: AnimationChannel[] = [],
        cameraNodeID?: number,
        weightTracks: WeightTrack[] = [],
        curves: SceneCurveDesc[] = [],
        /** From Scene.create: the gathered BVH geometry and the tree built off the main thread. */
        prebuilt?: { geometry: ReturnType<typeof Scene.collectBvhGeometry>; bvh?: BvhBuildResult },
    ) {
        this.cameraNodeID = cameraNodeID;
        this.sdfGrids = sdfGrids;
        // Vertices normalize to f32 up front (native holds f32 StaticVertexData):
        // vertex packing, the BVH build, and the scene cache then agree bit-exactly.
        Scene.roundVertices(meshes);
        // Geometry-less scenes are legal (pure-volume scenes like smoke.pyscene):
        // buffers pad to one zeroed struct and ray queries simply miss.
        this.hasEmissiveMaterials = materials.some((m) => m.header?.emissive ?? false);

        // Concatenate mesh geometry into global vertex/index buffers.
        const allVertices: StaticVertex[] = [];
        const allIndices: number[] = [];
        const meshDescs: MeshDescData[] = [];
        const instances: GeometryInstance[] = [];
        meshes.forEach((mesh, meshID) => {
            const vbOffset = allVertices.length;
            const ibOffset = allIndices.length;
            // A loop, not push(...): spreading a large mesh overflows the call stack.
            for (const v of mesh.vertices) allVertices.push(v);
            for (const i of mesh.indices) allIndices.push(i);
            meshDescs.push({
                vbOffset,
                ibOffset,
                vertexCount: mesh.vertices.length,
                indexCount: mesh.indices.length,
                // Prev-position buffer is laid out parallel to the vertex buffer.
                prevVbOffset: vbOffset,
                materialID: mesh.materialID,
            });
            // Materials with a displacement map turn the mesh into displaced
            // geometry (procedural AABBs; excluded from the triangle BVH).
            const displaced = materials[mesh.materialID]?.basic.texDisplacement !== undefined;
            if (displaced) this.hasDisplaced = true;
            instances.push({
                type: displaced ? GeometryType.DisplacedTriangleMesh : GeometryType.TriangleMesh,
                globalMatrixID: meshID,
                materialID: mesh.materialID,
                geometryID: meshID,
                vbOffset,
                ibOffset,
                instanceIndex: meshID,
                geometryIndex: 0,
                // IsDynamic routes getPrevPosW to the prevVertices buffer.
                flags: mesh.skin || mesh.morph ? 0x2 : 0,
            });
        });
        // The web scene flattens instances; native meshes are recovered from shared vertex data.
        const meshIDOf = new Map<StaticVertex[], number>();
        this.meshIDs = Uint32Array.from(meshes, (m) => {
            let id = meshIDOf.get(m.vertices);
            if (id === undefined) meshIDOf.set(m.vertices, (id = meshIDOf.size));
            return id;
        });
        this.meshMaterialIDs = meshDescs.map((d) => d.materialID);
        this.meshCounts = meshDescs.map((d) => ({ vertexCount: d.vertexCount, indexCount: d.indexCount }));
        this.vertexTotal = allVertices.length;
        this.triangleTotal = allIndices.length / 3;
        // HitInfo::init: the largest per-geometry primitive count sizes the primitive-index field.
        this.maxPrimitiveCount = Math.max(0, ...meshes.map((m) => m.indices.length / 3), ...curves.map((c) => c.indices.length));
        // SDF grid instances append after the triangle instances (they are
        // not in the triangle BVH; SBS/SVS use a separate primitive-AABB BVH).
        this.sdfInstanceFirst = instances.length;
        sdfGrids.forEach((desc, i) => {
            instances.push({
                type: GeometryType.SDFGrid,
                globalMatrixID: meshes.length + i,
                materialID: desc.materialID,
                geometryID: i,
                vbOffset: 0,
                ibOffset: 0,
                instanceIndex: meshes.length + i,
                geometryIndex: 0,
            });
        });
        // Curve instances append after SDF instances (not in the triangle BVH;
        // intersected via the curve segment-AABB path).
        this.curveInstanceFirst = instances.length;
        this.curveDescs = curves;
        let curveVbOffset = 0;
        let curveIbOffset = 0;
        curves.forEach((curve, i) => {
            instances.push({
                type: GeometryType.Curve,
                globalMatrixID: meshes.length + sdfGrids.length + i,
                materialID: curve.materialID,
                geometryID: i,
                vbOffset: curveVbOffset,
                ibOffset: curveIbOffset,
                instanceIndex: meshes.length + sdfGrids.length + i,
                geometryIndex: 0,
            });
            curveVbOffset += curve.positionsRadii.length / 4;
            curveIbOffset += curve.indices.length;
        });
        this.instanceCount = instances.length;
        this.drawList = meshDescs.map((m, i) => ({
            indexCount: m.indexCount,
            firstIndex: m.ibOffset,
            baseVertex: m.vbOffset,
            firstInstance: i,
        }));

        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const make = (name: string, data: ArrayBufferView | ArrayBuffer, structSize: number, extraFlags = ResourceBindFlags.None) => {
            let bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            // WebGPU forbids zero-size storage bindings: pad empty buffers to
            // one zeroed struct (counts gate all shader-side access).
            if (bytes.byteLength === 0) bytes = new Uint8Array(Math.max(structSize, 4));
            const buf = new Buffer(this.device, { size: bytes.byteLength, structSize, bindFlags: storage | extraFlags, memoryType: MemoryType.DeviceLocal, name: `Scene::${name}` });
            buf.setBlob(bytes);
            this.buffers[name] = buf;
            return buf;
        };

        make("vertices", packStaticVertices(allVertices), 48, ResourceBindFlags.Vertex);
        make("indices", new Uint32Array(allIndices), 4, ResourceBindFlags.Index);
        make("drawIDs", new Uint32Array(instances.map((inst) => inst.instanceIndex)), 4, ResourceBindFlags.Vertex);
        make("meshes", packMeshDescs(meshDescs), 32);
        make("geometryInstances", packGeometryInstances(instances), 32);

        // Node transforms: one node per mesh, then one per SDF grid instance
        // (globalMatrixID indexes this order). One merged buffer
        // (16-storage-buffer budget, same pattern as the BVH merge): world
        // matrices then inverse-transpose matrices.
        const nodeCount = meshes.length + sdfGrids.length + curves.length;
        const world = new Float32Array(nodeCount * 2 * 16);
        const putNode = (i: number, m: float4x4) => {
            world.set(m.toArray(), i * 16);
            world.set(transpose(inverse(m)).toArray(), (nodeCount + i) * 16);
        };
        meshes.forEach((mesh, i) => putNode(i, mesh.transform ?? float4x4.identity()));
        sdfGrids.forEach((desc, i) => putNode(meshes.length + i, desc.transform ?? float4x4.identity()));
        curves.forEach((desc, i) => putNode(meshes.length + sdfGrids.length + i, desc.transform ?? float4x4.identity()));
        make("worldMatrices", world, 64);
        this.invTransposeOffset = nodeCount;
        this.sdfInstanceBase = meshes.length;

        // Software RT BVH over world-space triangles (docs §5); displaced meshes use their own AABB region.
        const { bvhTris, displacedAabbs, displacedEntries } = prebuilt?.geometry ?? Scene.collectBvhGeometry(meshes, materials);
        const bvh = prebuilt?.bvh ?? buildBvh(bvhTris);

        // Whole-scene AABB = BVH root node bounds (nodes[0] = [min.xyz, _][max.xyz, _]).
        if (bvhTris.length > 0) {
            this.worldBounds = {
                min: [bvh.nodes[0]!, bvh.nodes[1]!, bvh.nodes[2]!],
                max: [bvh.nodes[4]!, bvh.nodes[5]!, bvh.nodes[6]!],
            };
        }
        // Curve segment-AABB BVH appended into the same merged buffer (16-
        // storage-buffer budget): prim entries encode (instance << 24 | segment).
        let curveBvhData: Float32Array = new Float32Array(0);
        this.sceneCurves = curves;
        if (curves.length > 0) {
            const built = buildCurveBvh(curves);
            curveBvhData = built.data;
            this.curveBvhOffset = 0; // patched after the triangle merge below
            this.curvePrimOffset = built.nodeWords / 4;
        }

        // Displaced-triangle AABB BVH rides in the same merged buffer.
        let displacedBvhData = new Float32Array(0);
        let displacedNodeWords = 0;
        if (displacedAabbs.length > 0) {
            const dbvh = buildAabbBvh(displacedAabbs);
            const encoded = new Uint32Array(dbvh.primIndices.length);
            for (let i = 0; i < dbvh.primIndices.length; i++) encoded[i] = displacedEntries[dbvh.primIndices[i]!]!;
            const primWords = Math.ceil(encoded.length / 4) * 4;
            displacedBvhData = new Float32Array(dbvh.nodes.length + primWords);
            displacedBvhData.set(dbvh.nodes, 0);
            new Uint32Array(displacedBvhData.buffer, dbvh.nodes.length * 4).set(encoded);
            displacedNodeWords = dbvh.nodes.length / 4;
        }

        // One merged buffer (16-storage-buffer budget): nodes then triangles.
        const bvhMerged = new Float32Array(bvh.nodes.length + bvh.tris.length + curveBvhData.length + displacedBvhData.length);
        bvhMerged.set(bvh.nodes, 0);
        bvhMerged.set(bvh.tris, bvh.nodes.length);
        this.bvhTrisOffset = bvh.nodes.length / 4;
        if (curveBvhData.length > 0) {
            bvhMerged.set(curveBvhData, bvh.nodes.length + bvh.tris.length);
            this.curveBvhOffset = (bvh.nodes.length + bvh.tris.length) / 4;
            this.curvePrimOffset += this.curveBvhOffset;
            this.curveBvhBytes = curveBvhData;
        }
        if (displacedBvhData.length > 0) {
            const base = bvh.nodes.length + bvh.tris.length + curveBvhData.length;
            bvhMerged.set(displacedBvhData, base);
            this.displacedBvhOffset = base / 4;
            this.displacedPrimOffset = this.displacedBvhOffset + displacedNodeWords;
        }
        // A scene animates if it has keyframe channels, morph-weight tracks, or
        // morph meshes (weights may be static-but-nonzero) — all rebuild per frame.
        const hasAnimation = ((animations.length > 0 || weightTracks.length > 0 || meshes.some((m) => m.morph)) && nodes.length > 0) || meshes.some((m) => m.vertexCache || m.polytubeCache) || curves.some((c) => c.vertexCache);
        if (hasAnimation) {
            // Animated scenes rebuild the BVH every frame; over-allocate to the
            // worst-case size (≤2N nodes + N tris) so animate() setBlobs in place —
            // destroying/recreating a buffer still referenced by an in-flight submit
            // is illegal.
            const numTris = allIndices.length / 3;
            // Curve caches rebuild their BVH too: reserve its worst case (2n nodes + prim words).
            const curveSegs = curves.reduce((n, c) => n + c.indices.length, 0);
            const curveWorst = curves.some((c) => c.vertexCache) ? (2 * curveSegs * 8 + Math.ceil(curveSegs / 4) * 4) * 4 : 0;
            this.bvhCapacityBytes = ((2 * numTris + 2) * 8 + numTris * 12) * 4 + Math.max(curveBvhData.byteLength, curveWorst);
            const buf = new Buffer(this.device, { size: Math.max(this.bvhCapacityBytes, 16), structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::bvhNodes" });
            buf.setBlob(bvhMerged);
            this.buffers["bvhNodes"] = buf;
        } else {
            make("bvhNodes", bvhMerged, 16);
        }

        // Retain source geometry + node graph for per-frame animation. Only for
        // animated scenes, so static scenes (e.g. Bistro) keep zero overhead.
        if (hasAnimation) {
            // Prev-frame buffers for motion vectors (static scenes alias the
            // current buffers in bindShaderData = zero motion).
            const prevWorld = new Buffer(this.device, { size: world.byteLength, structSize: 64, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::prevWorldMatrices" });
            prevWorld.setBlob(world);
            this.buffers["prevWorldMatrices"] = prevWorld;
            this.lastWorldMats = world;
            const prevVerts = packPrevVertices(allVertices);
            const prevBuf = new Buffer(this.device, { size: Math.max(prevVerts.byteLength, 16), structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::prevVertices" });
            prevBuf.setBlob(prevVerts);
            this.buffers["prevVertices"] = prevBuf;

            const allTimes = [...animations, ...weightTracks];
            const start = allTimes.reduce((s, ch) => Math.min(s, ch.times[0] ?? 0), Infinity);
            const duration = allTimes.reduce((d, ch) => Math.max(d, ch.times[ch.times.length - 1] ?? 0), 0);
            this.sourceMeshes = meshes;
            this.animData = { nodes, channels: animations, start: Number.isFinite(start) ? start : 0, duration, weightTracks };
            this.animNodeCount = nodeCount;
        }

        // Analytic lights.
        this.analyticLights = lights.map((l) => (l instanceof SceneLight ? l : new SceneLight(l, () => (this.lightsDirty = true))));
        this.lightCount = this.activeLights.length;
        make("lights", packLights(this.activeLights), 224);
        // Camera/light Animatable: recompute their pose from node globals each frame.
        this.hasAnimatedCameraOrLights = cameraNodeID !== undefined || lights.some((l) => l.nodeID !== undefined);
        this.addViewpoint(); // the default viewpoint (setCameraList redoes it for the scene's cameras)

        // Emissive geometry (LightCollection); inputs retained so runtime
        // emissive edits can rebuild the flux tables. materialDescs must be
        // assigned first — the rebuild reads it.
        this.lcMeshes = meshes;
        this.lcTextureManager = textureManager;
        materials = materials.map((m) => this.wrapMaterial(m));
        materials.forEach((m, i) => m.header?.emissive && this.emissiveMaterialIDs.add(i));
        this.materialDescs = materials;
        this.rebuildLightCollection();

        // Materials. Measured BRDFs carry bulk data (the MERL table plus its
        // albedo LUT); it all lands in the one shared material buffer, and each
        // material records its byte offsets (§9: no binding arrays in WGSL).
        this.materialCount = materials.length;
        this.materialDescs = materials;
        // Every measured material appends its arrays; offsets are handed to the shader.
        const regions: { at: number; data: ArrayBufferView }[] = [];
        let bufferSize = 0;
        const append = (data: ArrayBufferView) => {
            const at = bufferSize;
            regions.push({ at, data });
            // ByteAddressBuffer loads are 4-byte addressed, so every region starts aligned.
            bufferSize += (data.byteLength + 3) & ~3;
            return at;
        };
        const reserve = (floats: number) => {
            const at = bufferSize;
            bufferSize += floats * 4;
            return at;
        };
        materials.forEach((m, i) => {
            if (m.merl) {
                const data = append(m.merl.data);
                this.merlOffsets.set(i, { data, lut: reserve(kMERLAlbedoLUTSize * 4) });
            } else if (m.rgl) {
                const r = m.rgl;
                // RGL addresses its tables by element index (see packRGLMaterialBlob).
                const offsets: Record<string, number> = {};
                const put = (key: string, data: Float32Array) => (offsets[key] = append(data) / 4);
                put("theta", r.thetaI);
                put("phi", r.phiI);
                put("sigma", r.sigma);
                put("ndf", r.ndf);
                put("vndf", r.vndf);
                put("lumi", r.luminance);
                put("rgb", r.rgb);
                put("vndfMarginal", r.vndfMarginal);
                put("lumiMarginal", r.lumiMarginal);
                put("vndfConditional", r.vndfConditional);
                put("lumiConditional", r.lumiConditional);
                offsets["albedoLUT"] = reserve(kRGLAlbedoLUTSize * 4) / 4;
                this.rglOffsets.set(i, offsets);
            } else if (m.merlMix) {
                const mix = m.merlMix;
                if (mix.brdfs.length === 0) throw new RuntimeError("MERLMix material has no BRDFs");
                // The BRDFs sit back to back at a common stride, as MERLMixMaterial.cpp checks.
                const stride = mix.brdfs[0]!.data.byteLength;
                let dataOffset = -1;
                for (const brdf of mix.brdfs) {
                    if (brdf.data.byteLength !== stride) throw new RuntimeError("MERLMix: every BRDF must have the same sample count");
                    const at = append(brdf.data);
                    if (dataOffset < 0) dataOffset = at;
                }
                // Per-BRDF fitted approximation used for sampling.
                const extra = new Uint8Array(mix.brdfs.length * kDiffuseSpecularDataSize);
                const extraView = new DataView(extra.buffer);
                mix.brdfs.forEach((b, k) => writeDiffuseSpecularData(extraView, k * kDiffuseSpecularDataSize, b.extraData));
                // Index map: width, height, then one byte per texel (§9: point-sampled
                // indices cannot go through the shared linear material sampler).
                const map = mix.indexMap;
                const mapBytes = new Uint8Array(8 + ((map.indices.length + 3) & ~3));
                const mapView = new DataView(mapBytes.buffer);
                mapView.setUint32(0, map.width, true);
                mapView.setUint32(4, map.height, true);
                mapBytes.set(map.indices, 8);
                this.merlMixOffsets.set(i, {
                    data: dataOffset,
                    stride,
                    extra: append(extra),
                    indexMap: append(mapBytes),
                    // One 256-entry float4 LUT per BRDF, stacked (upstream's 2D texture).
                    lut: reserve(kMERLAlbedoLUTSize * 4 * mix.brdfs.length),
                });
            }
        });
        const materialBuffer = new Uint8Array(Math.max(bufferSize, 16));
        for (const r of regions) materialBuffer.set(new Uint8Array(r.data.buffer, r.data.byteOffset, r.data.byteLength), r.at);
        const blobBytes = new Uint8Array(materials.length * 128);
        materials.forEach((m, i) => {
            this.materialTypes.add(
                m.merl ? MaterialType.MERL : m.rgl ? MaterialType.RGL : m.merlMix ? MaterialType.MERLMix : (m.header?.materialType ?? MaterialType.Standard),
            );
            blobBytes.set(this.packMaterial(m, i), i * 128);
        });
        make("materialData", blobBytes, 128);
        make("materialBuffer0", materialBuffer, 4);
        make("curveDummy", new Uint32Array(16), 32); // StaticCurveVertexData-sized dummy
        if (curves.length > 0) {
            const { cv, ci, cd } = packCurves(curves);
            make("curveVertices", cv, 32);
            make("curveIndices", ci, 4);
            make("curves", cd, 24);
        }
        make("gridVolumeDummy", new Uint32Array(64), 256); // GridVolumeData-sized dummy (2x float4x4 + params)

        // Material textures in per-format/size arrays (docs §6.2).
        this.buildMaterialTextures(textureManager);
        this.dummyTexture = this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0, 0, 0, 0]));
        // Standalone displacement texture (v1: one displaced material per scene).
        const dispHandle = materials.map((m) => m.basic.texDisplacement).find((h) => h !== undefined);
        if (dispHandle !== undefined) {
            const source = textureManager.getSource(dispHandle & 0x1fffffff);
            if (source) {
                const tex = this.device.createTexture2D(
                    source.bitmap.width,
                    source.bitmap.height,
                    ResourceFormat.RGBA8Unorm,
                    1,
                    1,
                    undefined,
                    ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
                );
                this.device.gpuDevice.queue.copyExternalImageToTexture({ source: source.bitmap }, { texture: tex.gpuTexture }, [source.bitmap.width, source.bitmap.height]);
                this.displacementTexture = tex;
            }
        }
        this.texture3D = this.device.createTexture3D(1, 1, 1, ResourceFormat.RGBA32Float, 1);
        this.gridRangeTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.RG32Float, 1);
        this.gridIndirectionTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.RGBA32Uint, 1);
        this.gridAtlasTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.R32Float, 1);
        // Mirrors MaterialSystem's default texture sampler: trilinear + anisotropy 8.
        this.sampler = this.device.createSampler({ maxAnisotropy: 8 });
    }

    private gridRangeTex: Texture;
    private gridIndirectionTex: Texture;
    private gridAtlasTex: Texture;

    /** Uploads the texture manager's textures as per-format/size arrays (docs §6.2). */
    private buildMaterialTextures(textureManager: TextureManager): void {
        const packed = textureManager.build(this.device);
        this.textureBuckets = packed.buckets.map((b) => b.texture);
        // Mip chains for texture-LOD (uploads are queue-ordered before the blits); BC arrays carry theirs.
        for (const b of packed.buckets) if (b.generateMips) b.texture.generateMips(this.device.renderContext);

        this.textureCount = Math.max(textureManager.count, 1);
        this.builtTextureCount = textureManager.count;
        // 1-row texture (16-storage-buffer budget: frees a slot in every scene-bound kernel).
        this.texInfoTexture = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: packed.texInfo.length / 4,
            height: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "Scene::materialTextureUvScale",
        });
        this.texInfoTexture.setSubresourceBlob(0, 0, new Uint8Array(packed.texInfo.buffer, packed.texInfo.byteOffset, packed.texInfo.byteLength));
    }

    /** The material system's texture manager (Scene::getMaterialSystem().getTextureManager()). */
    get textureManager(): TextureManager {
        return this.lcTextureManager;
    }

    /**
     * Mirrors Scene::replaceMaterial (MaterialSystem::replaceMaterial): material `index` becomes
     * `desc`, whose textures must already be in `textureManager`. Returns true when the scene
     * defines changed (a new material type or texture count), i.e. passes must recompile.
     */
    replaceMaterial(index: number, desc: SceneMaterialDesc): boolean {
        if (!(index >= 0 && index < this.materialDescs.length)) throw new RuntimeError("Material ID is invalid.");
        if (desc.merl || desc.rgl || desc.merlMix) throw new RuntimeError("Scene.replaceMaterial: measured materials can't replace at runtime (their data lives in the shared material buffer)");
        const definesBefore = this.getSceneDefines().key();
        this.materialDescs[index] = this.wrapMaterial(desc);
        if (this.lcTextureManager.count !== this.builtTextureCount) this.buildMaterialTextures(this.lcTextureManager);
        this.materialTypes = new Set(this.materialDescs.map((m) => (m.merl ? MaterialType.MERL : m.rgl ? MaterialType.RGL : m.merlMix ? MaterialType.MERLMix : (m.header?.materialType ?? MaterialType.Standard))));
        this.materialDescs.forEach((m, i) => this.buffers["materialData"]!.setBlob(this.packMaterial(m, i), i * 128));
        this.rebuildLightCollection();
        return this.getSceneDefines().key() !== definesBefore;
    }

    /** Mirrors Scene::setEnvMap; python's setEnvMap(path) (Scene::loadEnvMap) loads asynchronously. */
    setEnvMap(envMap: EnvMap | string | null): boolean | void {
        if (typeof envMap !== "string") {
            this.envMap = envMap;
            return;
        }
        const path = envMap;
        this.pendingEnvMap = (async () => {
            const { EnvMap } = await import("./Lights/EnvMap.js");
            this.envMap = await EnvMap.createFromUrl(this.device, await resolveAssetUrl(path, "", AssetCategory.Any));
        })().catch((err) => Logger.warning(`Failed to load environment map from '${path}': ${err}`));
        return true;
    }
    /** The envMap load a python setEnvMap(path) started (awaitable by callers). */
    pendingEnvMap: Promise<void> | null = null;

    /** Mirrors Scene::setCameraBounds; the camera controller clamps its position to the box. */
    setCameraBounds(minPoint: { x: number; y: number; z: number }, maxPoint: { x: number; y: number; z: number }): void {
        this.cameraBounds = new AABB(minPoint, maxPoint);
    }
    cameraBounds: AABB | null = null;

    /** Python `scene.memory_usage` (Scene::getMemoryUsageInBytes: SceneStats::getTotalMemory). */
    get memory_usage(): number {
        const sum = (o: object): number => Object.entries(o).reduce((n, [k, v]) => n + (typeof v === "object" && v ? sum(v) : k.endsWith("MemoryInBytes") ? Number(v) : 0), 0);
        return sum(this.getSceneStats());
    }

    /** Python `scene.get_material(index | name)`. */
    get_material(ref: number | string): SceneMaterialDesc {
        return this.getMaterial(ref);
    }

    /** Mirrors Scene::getGeometryIDs(material): global geometry IDs (meshes, curves, SDF grids) using it. */
    getGeometryIDsForMaterial(material: SceneMaterialDesc | number): number[] {
        const id = typeof material === "number" ? material : this.materialDescs.indexOf(material);
        const ids: number[] = [];
        const meshCount = this.meshIDs.length ? Math.max(...this.meshIDs) + 1 : 0;
        const seen = new Set<number>();
        this.meshMaterialIDs.forEach((m, i) => {
            const meshID = this.meshIDs[i]!;
            if (m === id && !seen.has(meshID)) (seen.add(meshID), ids.push(meshID));
        });
        ids.sort((a, b) => a - b);
        this.curveDescs.forEach((c, i) => c.materialID === id && ids.push(meshCount + i));
        this.sdfGrids.forEach((g, i) => g.materialID === id && ids.push(meshCount + this.curveDescs.length + i));
        return ids;
    }

    /**
     * Mirrors Scene::getGeometryUVTiles (createMeshUVTiles): per unit UV square, the bounds of the
     * triangles inside it, plus one tile for triangles spanning squares (which absorbs contained tiles).
     */
    getGeometryUVTiles(geometryID: number): Rectangle[] {
        const cached = this.uvTiles.get(geometryID);
        if (cached) return cached;
        const i = this.meshIDs.indexOf(geometryID);
        if (i < 0) return [];
        const { vertices, indices } = this.lcMeshes[i]!;
        const large = new Rectangle();
        const tiles = new Map<string, { key: [number, number]; tile: Rectangle }>();
        for (let t = 0; t + 2 < indices.length; t += 3) {
            const uv = [0, 1, 2].map((k) => vertices[indices[t + k]!]!.texCrd);
            const cells = uv.map((c) => [Math.floor(c.x), Math.floor(c.y)] as [number, number]);
            const same = cells.every((c) => c[0] === cells[0]![0] && c[1] === cells[0]![1]);
            let tile = large;
            if (same) {
                const id = cells[0]!.join();
                const entry = tiles.get(id) ?? { key: cells[0]!, tile: new Rectangle() };
                tiles.set(id, entry);
                tile = entry.tile;
            }
            for (const c of uv) tile.include(c);
        }
        // std::map<int2> order: by x, then y.
        const sorted = [...tiles.values()].sort((a, b) => a.key[0] - b.key[0] || a.key[1] - b.key[1]);
        const result = sorted.filter((e) => !large.contains(e.tile)).map((e) => e.tile);
        if (large.valid) result.push(large);
        this.uvTiles.set(geometryID, result);
        return result;
    }
    private uvTiles = new Map<number, Rectangle[]>();

    /** Python `scene.get_mesh(mesh_id)` (MeshDesc vertex_count / triangle_count). */
    get_mesh(meshID: number): { vertex_count: number; triangle_count: number } {
        const i = this.meshIDs.indexOf(meshID);
        if (i < 0) throw new RuntimeError(`Scene.get_mesh: no mesh ${meshID}`);
        const d = this.meshCounts[i]!;
        return { vertex_count: d.vertexCount, triangle_count: d.indexCount / 3 };
    }

    /** Mirrors Scene::getLightCount. */
    getLightCount(): number {
        return this.analyticLights.length;
    }

    /** Mirrors Scene::getLight / getLightByName (live object; call updateLights() after edits). */
    getLight(ref: number | string): AnalyticLight {
        const light = typeof ref === "number" ? this.analyticLights[ref] : this.analyticLights.find((l) => l.name === ref);
        if (!light) throw new RuntimeError(`Scene.getLight: no light '${ref}'`);
        return light;
    }

    /** Re-packs analytic lights after runtime property edits (mirrors Light change tracking in Scene::update). */
    updateLights(): void {
        this.lightsDirty = false;
        const active = this.activeLights;
        this.lightCount = active.length;
        this.buffers["lights"]!.setBlob(packLights(active));
    }

    /** Mirrors Scene::getActiveLights: the lights in the light buffer, in its order. */
    get activeLights(): AnalyticLight[] {
        return this.analyticLights.filter((l) => (l as SceneLight).active !== false);
    }

    /** Python `scene.bounds` (Scene::getSceneBounds): the world-space geometry bounds. */
    get bounds(): AABB {
        const b = this.worldBounds;
        return b ? new AABB({ x: b.min[0], y: b.min[1], z: b.min[2] }, { x: b.max[0], y: b.max[1], z: b.max[2] }) : new AABB();
    }

    /** Mirrors Scene::getGridVolume / getGridVolumeByName (python also `getVolume`). */
    getGridVolume(ref: number | string): import("./Volume/GridVolume.js").GridVolume | undefined {
        return typeof ref === "number" ? this.gridVolumes[ref] : this.gridVolumes.find((v) => v.name === ref);
    }
    getVolume(ref: number | string): import("./Volume/GridVolume.js").GridVolume | undefined {
        return this.getGridVolume(ref);
    }
    /** Python `scene.volumes` (deprecated alias of gridVolumes). */
    get volumes(): import("./Volume/GridVolume.js").GridVolume[] {
        return this.gridVolumes;
    }

    /** Python `scene.lights` (Scene::getLights). */
    get lights(): AnalyticLight[] {
        return this.analyticLights;
    }

    /** Set by SceneLight edits; the buffer is repacked before the next bind. */
    private lightsDirty = false;

    /** A material record as a SceneMaterial (native property names; edits repack it). */
    private wrapMaterial(m: SceneMaterialDesc): SceneMaterial {
        return m instanceof SceneMaterial ? m : new SceneMaterial(m, (self) => this.updateMaterial(self));
    }

    /** Python `scene.materials` (Scene::getMaterials). */
    get materials(): SceneMaterialDesc[] {
        return this.materialDescs;
    }

    /** Mirrors Scene::getMaterial / getMaterialByName (live descriptor; call updateMaterial() after edits). */
    getMaterial(ref: number | string): SceneMaterialDesc {
        const mat = typeof ref === "number" ? this.materialDescs[ref] : this.materialDescs.find((m) => m.name === ref);
        if (!mat) throw new RuntimeError(`Scene.getMaterial: no material '${ref}'`);
        return mat;
    }

    /** Re-packs one material blob after runtime property edits (mirrors
     *  MaterialSystem::update); emissive edits also rebuild the
     *  LightCollection flux tables. */
    updateMaterial(ref: number | string | SceneMaterialDesc): void {
        const index = typeof ref === "object" ? this.materialDescs.indexOf(ref) : typeof ref === "number" ? ref : this.materialDescs.findIndex((m) => m.name === ref);
        const m = this.materialDescs[index];
        if (!m) throw new RuntimeError(`Scene.updateMaterial: no material '${String(ref)}'`);
        this.buffers["materialData"]!.setBlob(this.packMaterial(m, index), index * 128);
        // Emissive edits change the NEE flux distribution (mirrors native
        // MaterialsChanged handling). Presence toggles that flip scene defines
        // still require pass recreation by the caller.
        if (m.header?.emissive || this.emissiveMaterialIDs.has(index)) this.rebuildLightCollection();
        if (m.header?.emissive) this.emissiveMaterialIDs.add(index);
        else this.emissiveMaterialIDs.delete(index);
    }
    /** Materials emissive at their last update (an edit turning emission off must rebuild too). */
    private emissiveMaterialIDs = new Set<number>();

    /** Packs one material blob; the alpha mode follows native updateAlphaMode unless given explicitly. */
    private packMaterial(m: SceneMaterialDesc, index: number): Uint8Array {
        const header: MaterialHeaderDesc = { materialType: MaterialType.Standard, ...m.header };
        if (m.merl) {
            const offsets = this.merlOffsets.get(index)!;
            return packMERLMaterialBlob(header, { dataOffset: offsets.data, albedoLUTOffset: offsets.lut, extraData: m.merl.extraData });
        }
        if (m.merlMix) {
            const o = this.merlMixOffsets.get(index)!;
            return packMERLMixMaterialBlob(header, {
                brdfCount: m.merlMix.brdfs.length,
                byteStride: o.stride,
                dataOffset: o.data,
                extraDataOffset: o.extra,
                indexMapOffset: o.indexMap,
                albedoLUTOffset: o.lut,
                texNormalMap: m.merlMix.texNormalMap,
            });
        }
        if (m.rgl) {
            const o = this.rglOffsets.get(index)!;
            return packRGLMaterialBlob(header, {
                phiSize: m.rgl.phiI.length,
                thetaSize: m.rgl.thetaI.length,
                sigmaSize: m.rgl.sigmaSize,
                ndfSize: m.rgl.ndfSize,
                vndfSize: m.rgl.vndfSize,
                lumiSize: m.rgl.lumiSize,
                offsets: o as unknown as Parameters<typeof packRGLMaterialBlob>[1]["offsets"],
            });
        }
        if (header.alphaMode === undefined) header.alphaMode = this.deriveAlphaMode(header, m.basic);
        if (header.deltaSpecular === undefined) header.deltaSpecular = isDeltaSpecularStandard(header, m.basic);
        return packBasicMaterialBlob(header, m.basic);
    }

    /**
     * Mirrors MERLFile::computeAlbedoLUT: integrates each measured BRDF over the
     * hemisphere at cosTheta = (1..N)/N and writes the table into the material
     * buffer. Native precomputes this per material (and caches it as a `.dds`);
     * the web integrates the live scene, which is the same BSDFIntegrator run.
     */
    async computeMeasuredAlbedoLUTs(ctx: RenderContext): Promise<void> {
        const targets: [number, number][] = [
            ...[...this.merlOffsets].map(([id, o]) => [id, o.lut] as [number, number]),
            ...[...this.rglOffsets].map(([id, o]) => [id, o["albedoLUT"]! * 4] as [number, number]),
        ];
        if (targets.length === 0 && this.merlMixOffsets.size === 0) return;
        const { BSDFIntegrator } = await import("../Rendering/Materials/BSDFIntegrator.js");
        const size = kMERLAlbedoLUTSize; // MERL and RGL both use 256
        const cosThetas = Array.from({ length: size }, (_v, i) => (i + 1) / size);
        /** Integrates one material of `scene` and writes the 256-entry float4 table. */
        const writeLUT = async (integrator: InstanceType<typeof BSDFIntegrator>, materialID: number, byteOffset: number) => {
            const albedos = await integrator.integrateIsotropic(ctx, materialID, cosThetas);
            const lut = new Float32Array(size * 4);
            albedos.forEach((a, i) => lut.set([a.x, a.y, a.z, 1], i * 4));
            this.buffers["materialBuffer0"]!.setBlob(new Uint8Array(lut.buffer), byteOffset);
        };

        if (targets.length > 0) {
            const integrator = new BSDFIntegrator(this.device, this);
            for (const [materialID, byteOffset] of targets) await writeLUT(integrator, materialID, byteOffset);
        }

        // MERLMix stacks one table per BRDF. Each row is that BRDF's own albedo,
        // which native gets from a dummy scene holding the single MERL material
        // (MERLFile::computeAlbedoLUT); one temporary scene per mix does the same.
        for (const [materialID, offsets] of this.merlMixOffsets) {
            const mix = this.materialDescs[materialID]!.merlMix!;
            const temp = new Scene(
                this.device,
                [],
                mix.brdfs.map((merl) => ({ name: merl.name, basic: {}, merl, header: { materialType: MaterialType.MERL } })),
            );
            const integrator = new BSDFIntegrator(this.device, temp);
            for (let k = 0; k < mix.brdfs.length; k++) await writeLUT(integrator, k, offsets.lut + k * size * 16);
        }
    }

    /**
     * Mirrors BasicMaterial::updateAlphaMode: alpha testing is enabled only when the
     * base color alpha can fall below the threshold — the texture's alpha range
     * (TextureAnalyzer parity via a CPU scan) when textured, else the constant alpha.
     * Only StandardMaterial has an alpha channel in its base color slot.
     */
    private deriveAlphaMode(header: MaterialHeaderDesc, basic: BasicMaterialDesc): AlphaMode {
        if (header.materialType !== MaterialType.Standard) return AlphaMode.Opaque;
        const threshold = header.alphaThreshold ?? 0.5;
        let minAlpha = basic.baseColor?.w ?? 1;
        const tex = basic.texBaseColor;
        if (tex !== undefined && ((tex >>> 29) & 0x3) === TextureHandleMode.Texture) {
            // Native assumes the full [0,1] range until the texture is analyzed.
            minAlpha = this.lcTextureManager.getAlphaRange(tex & 0x1fffffff)?.[0] ?? 0;
        }
        return minAlpha < threshold ? AlphaMode.Mask : AlphaMode.Opaque;
    }

    /** (Re)builds the LightCollection buffers (mirrors Scene::updateLights on
     *  MaterialsChanged: emissive flux tables follow runtime material edits).
     *  Old buffers are replaced, not destroyed — in-flight submits stay valid. */
    private rebuildLightCollection(): void {
        const lc = buildLightCollection(
            this.lcMeshes,
            this.materialDescs.map((m) => {
                const em = m.basic.emissive;
                const factor = m.basic.emissiveFactor ?? 1;
                // Textured emissives integrate per triangle (EmissiveIntegrator
                // semantics); texture handle low bits = textureID (mode bits high).
                const texEmissive = m.basic.texEmissive;
                const emissiveTexture =
                    texEmissive !== undefined ? (this.lcTextureManager.readLinearPixels(texEmissive & 0x1fffffff) ?? undefined) : undefined;
                return {
                    emissive: m.header?.emissive ?? false,
                    radiance: [(em?.x ?? 0) * factor, (em?.y ?? 0) * factor, (em?.z ?? 0) * factor] as [number, number, number],
                    emissiveTexture,
                    emissiveFactor: factor,
                };
            }),
        );
        this.emissiveTriangleCount = lc.triangleCount;
        this.emissiveActiveTriangleCount = lc.activeTriangles.length;
        this.emissiveMeshCount = lc.meshCount;
        this.emissiveMeshData = lc.meshData;
        this.emissiveFluxes = new Float32Array(lc.triangleCount);
        const fluxView = new DataView(lc.fluxData);
        for (let i = 0; i < lc.triangleCount; i++) this.emissiveFluxes[i] = fluxView.getFloat32(i * 32, true);
        // Retain builder inputs for the LightBVH sampler (native builds from the
        // UNPACKED emissive triangles: octahedral-decoded normals).
        const triView = new DataView(lc.triangleData);
        this.emissiveTriangles = Array.from({ length: lc.triangleCount }, (_v, i) => ({
            posW: [0, 1, 2].map((k) => [
                triView.getFloat32(i * 64 + k * 16, true),
                triView.getFloat32(i * 64 + k * 16 + 4, true),
                triView.getFloat32(i * 64 + k * 16 + 8, true),
            ]) as [Vec3, Vec3, Vec3],
            normal: decodeNormal2x16Host(triView.getUint32(i * 64 + 48, true)),
            flux: this.emissiveFluxes[i]!,
        }));
        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const remake = (name: string, data: ArrayBufferView | ArrayBuffer, structSize: number) => {
            let bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            if (bytes.byteLength === 0) bytes = new Uint8Array(Math.max(structSize, 4));
            const buf = new Buffer(this.device, { size: bytes.byteLength, structSize, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: `Scene::${name}` });
            buf.setBlob(bytes);
            this.buffers[name] = buf;
        };
        remake("emissiveTriangles", lc.triangleData, 64);
        remake("emissiveFlux", lc.fluxData, 32);
        remake("emissiveActiveTriangles", lc.activeTriangles, 4);
        remake("emissiveTriToActive", lc.triToActiveMapping, 4);
        remake("emissiveMeshData", lc.meshData, 16);
        remake("emissivePerMeshInstanceOffset", lc.perMeshInstanceOffset, 4);
        this.emissiveVersion++;
    }

    /** Python `scene.updateCallback`: called as (scene, time) at the start of every frame's update (Scene::update). */
    updateCallback: ((scene: Scene, time: number) => void) | null = null;
    /** Runs updateCallback for this frame; the frame loops call it before animate(). */
    runUpdateCallback(time: number): void {
        try {
            this.updateCallback?.(this, time);
        } catch (e) {
            // A failing script callback is dropped (logged once) rather than stopping every frame.
            Logger.error(`Scene.updateCallback failed and was removed: ${String(e).split("\n").slice(-2).join(" ")}`);
            this.updateCallback = null;
        }
    }

    /** Mirrors Scene::hasAnimation: the scene has keyframe animations. */
    hasAnimation(): boolean {
        return this.animData !== null;
    }

    /** Mirrors Scene::isAnimated: it has animations and they are enabled (setIsAnimated). */
    isAnimated(): boolean {
        return this.hasAnimation() && this.animationEnabled;
    }

    /** Mirrors Scene::setIsAnimated (python `scene.animated`): pauses or resumes the animations. */
    setIsAnimated(animated: boolean): void {
        this.animationEnabled = animated;
    }
    get animated(): boolean {
        return this.animationEnabled;
    }
    set animated(v: boolean) {
        this.setIsAnimated(v);
    }

    private animationEnabled = true;

    /** Mirrors Scene::setIsLooped (python `scene.loopAnimations`): time wraps at the longest animation. */
    loopAnimations = true;
    setIsLooped(looped: boolean): void {
        this.loopAnimations = looped;
    }
    isLooped(): boolean {
        return this.loopAnimations;
    }
    /** Mirrors Scene::setCameraSpeed (the camera controller's speed). */
    cameraSpeed = 1;
    /** Mirrors Scene::getMetadata (camera and render settings from the imported asset). */
    metadata: SceneMetadata = {};

    /** Mirrors Scene::getScript: render settings, animation state, camera selection and pose, camera speed. */
    getScript(sceneVar: string): string {
        let c = "";
        for (const [k, v] of Object.entries(this.renderSettings)) c += ScriptWriter.makeSetProperty(`${sceneVar}.renderSettings`, k, v);
        if (this.hasAnimation() && !this.animationEnabled) c += ScriptWriter.makeSetProperty(sceneVar, "animated", false);
        if (this.activeCameraIndex !== 0) c += `${sceneVar}.camera = ${sceneVar}.cameras[${this.activeCameraIndex}]\n`;
        c += this.camera.getScript(`${sceneVar}.camera`);
        c += ScriptWriter.makeSetProperty(sceneVar, "cameraSpeed", this.cameraSpeed);
        return c;
    }

    /**
     * Advances keyframe animation to `timeSec` (looped over the clip duration),
     * re-uploading node world matrices and rebuilding the software-RT BVH over the
     * new world-space triangles. Returns true if the scene animated (so the caller
     * can reset accumulation); no-op for static scenes and while animations are
     * disabled (setIsAnimated). The software-RT BVH is refit (see refitBvh).
     */
    animate(timeSec: number): boolean {
        if (!this.animData || !this.sourceMeshes || !this.animationEnabled) return false;
        const meshes = this.sourceMeshes;
        // Mirrors AnimationController: loop raw time over the clip length.
        // Clips needn't start at t=0 (e.g. FBX): before the first key the
        // samplers clamp to it (native Constant pre-behavior).
        const sampleTime = this.animData.duration > 0 ? (this.loopAnimations ? timeSec % this.animData.duration : timeSec) : 0;
        const globals = evaluateGlobals(this.animData, sampleTime);
        // AnimationController: vertex caches take the looped time, or the raw time without node animations,
        // and cycle before their first sample when they are shorter than the node animations.
        const cacheTime = this.animData.duration > 0 ? sampleTime : timeSec;
        const cacheLength = meshes.reduce((d, m) => Math.max(d, m.vertexCache?.times.at(-1) ?? 0), 0);
        const cachePreCycle = cacheLength < this.animData.duration;
        // Curves (LSS and poly-tube) loop over their own length.
        const curveLength = Math.max(
            this.sceneCurves.reduce((d, c) => Math.max(d, c.vertexCache?.times.at(-1) ?? 0), 0),
            meshes.reduce((d, m) => Math.max(d, m.polytubeCache?.times.at(-1) ?? 0), 0),
        );
        const curveTime = curveLength > 0 && this.loopAnimations ? cacheTime % curveLength : cacheTime;

        // Animatable camera/lights: rederive their pose from the node globals
        // (glTF cameras/lights aim down local -Z; up is local +Y).
        if (this.hasAnimatedCameraOrLights) this.updateAnimatedCameraAndLights(globals);

        // Per mesh: current world matrix + world-space vertex positions. Skinned
        // meshes deform to world space on the CPU (identity world matrix, skinned
        // verts written into the shared vertex buffer); rigid meshes ride their
        // node's global matrix (object-space verts unchanged, matrix updated).
        const worldMats: float4x4[] = [];
        const worldPos: float3[][] = [];
        // LightCollection inputs for this frame: deformed verts ride an identity matrix.
        const lcInputs: SceneMeshDesc[] = [];
        let emissiveChanged = false;
        let vbOffset = 0;
        for (const [meshID, mesh] of meshes.entries()) {
            // Morph (blend shapes) deform the bind pose first (glTF applies morph
            // before skinning); the morphed object-space verts feed skin or matrix.
            const base = mesh.polytubeCache
                ? posePolytube(mesh, curveTime, curveLength < this.animData.duration)
                : mesh.vertexCache
                ? sampleVertexCache(mesh.vertexCache, cacheTime, cachePreCycle, mesh.vertices, this.loopAnimations)
                : mesh.morph
                  ? applyMorph(mesh.vertices, mesh.morph, sampleMorphWeights(mesh.morph, this.animData.weightTracks, sampleTime))
                  : mesh.vertices;
            const isEmissive = this.materialDescs[mesh.materialID]?.header?.emissive === true;
            if (mesh.skin) {
                const skinned = skinVertices(base, mesh.skin, computeSkinMatrices(mesh.skin, globals));
                this.rollPrevVertices(meshID, vbOffset, skinned);
                this.buffers["vertices"]!.setBlob(packStaticVertices(skinned), vbOffset * 48);
                worldMats.push(float4x4.identity());
                worldPos.push(skinned.map((v) => v.position));
                lcInputs.push({ ...mesh, vertices: skinned, transform: undefined });
                if (isEmissive) emissiveChanged = true; // deformed every frame
            } else {
                const m = mesh.nodeID !== undefined && globals[mesh.nodeID] ? globals[mesh.nodeID]! : (mesh.transform ?? float4x4.identity());
                // Morphed / vertex-cached non-skinned meshes: re-upload deformed object-space verts.
                if (mesh.morph || mesh.vertexCache || mesh.polytubeCache) {
                    this.rollPrevVertices(meshID, vbOffset, base);
                    this.buffers["vertices"]!.setBlob(packStaticVertices(base), vbOffset * 48);
                    if (isEmissive) emissiveChanged = true;
                }
                worldMats.push(m);
                worldPos.push(base.map((v) => transformPoint(m, v.position)));
                lcInputs.push({ ...mesh, vertices: base, transform: m });
                if (isEmissive) {
                    // Mirrors LightCollection::update's isMatrixChanged check per mesh light.
                    const cur = Float32Array.from(m.toArray());
                    // First frame compares against the build-time transform (the LightCollection's state).
                    const prev = this.lastEmissiveMats.get(meshID) ?? (mesh.transform ?? float4x4.identity()).toArray();
                    if (prev.some((v, i) => v !== cur[i])) emissiveChanged = true;
                    this.lastEmissiveMats.set(meshID, cur);
                }
            }
            vbOffset += mesh.vertices.length;
        }
        // Mirrors LightCollection::update (UpdateTriangleVertices): emissive triangles follow
        // their animated instances; samplers refit/rebuild off emissiveVersion.
        if (emissiveChanged && this.hasEmissiveMaterials) {
            this.lcMeshes = lcInputs;
            this.rebuildLightCollection();
        }

        // Rebuild the worldMatrices buffer (world + inverse-transpose halves) in place.
        const nodeCount = this.animNodeCount;
        const world = new Float32Array(nodeCount * 2 * 16);
        const putNode = (i: number, m: float4x4) => {
            world.set(m.toArray(), i * 16);
            world.set(transpose(inverse(m)).toArray(), (nodeCount + i) * 16);
        };
        meshes.forEach((_m, i) => putNode(i, worldMats[i]!));
        this.sdfGrids.forEach((desc, i) => putNode(meshes.length + i, desc.transform ?? float4x4.identity()));
        this.sceneCurves.forEach((desc, i) => putNode(meshes.length + this.sdfGrids.length + i, desc.transform ?? float4x4.identity()));
        // Prev matrices = the ones used last frame (mirrors Scene::updateMatrices).
        if (this.lastWorldMats && this.buffers["prevWorldMatrices"]) {
            this.buffers["prevWorldMatrices"]!.setBlob(this.lastWorldMats);
        }
        this.lastWorldMats = world;
        this.buffers["worldMatrices"]!.setBlob(world);

        // Rebuild the world-space triangle BVH over the current (rigid/skinned) verts.
        const bvhTris: BvhTriangle[] = [];
        meshes.forEach((mesh, meshID) => {
            const wp = worldPos[meshID]!;
            for (let p = 0; p < mesh.indices.length / 3; p++) {
                bvhTris.push({
                    v0: wp[mesh.indices[p * 3]!]!,
                    v1: wp[mesh.indices[p * 3 + 1]!]!,
                    v2: wp[mesh.indices[p * 3 + 2]!]!,
                    instanceIndex: meshID,
                    primitiveIndex: p,
                });
            }
        });
        // Refit the previous frame's BVH (rebuilt when the refit degrades; see refitBvh).
        const bvh = this.animatedBvh ? refitBvh(this.animatedBvh, bvhTris) : buildBvh(bvhTris);
        this.animatedBvh = bvh;
        if (bvhTris.length > 0) {
            this.worldBounds = { min: [bvh.nodes[0]!, bvh.nodes[1]!, bvh.nodes[2]!], max: [bvh.nodes[4]!, bvh.nodes[5]!, bvh.nodes[6]!] };
        }
        // Curve vertex caches (AnimatedVertexCache's curves): positions lerp, radii stay; curves loop
        // over their own length and hold after the last sample; their BVH is rebuilt.
        if (this.sceneCurves.some((c) => c.vertexCache)) {
            const posed = this.sceneCurves.map((c) => (c.vertexCache ? { ...c, positionsRadii: sampleCurveCache(c, curveTime, curveLength < (this.animData?.duration ?? 0)) } : c));
            this.buffers["curveVertices"]!.setBlob(packCurves(posed).cv);
            const built = buildCurveBvh(posed);
            this.curveBvhBytes = built.data;
            this.curvePrimOffset = this.curveBvhOffset + built.nodeWords / 4;
        }
        const curveExtra = this.curveBvhBytes ?? new Float32Array(0);
        const bvhMerged = new Float32Array(bvh.nodes.length + bvh.tris.length + curveExtra.length);
        bvhMerged.set(bvh.nodes, 0);
        bvhMerged.set(bvh.tris, bvh.nodes.length);
        this.bvhTrisOffset = bvh.nodes.length / 4;
        if (curveExtra.length > 0) {
            // Static curve BVH rides after the rebuilt triangle data.
            bvhMerged.set(curveExtra, bvh.nodes.length + bvh.tris.length);
            const nodeWords = this.curvePrimOffset - this.curveBvhOffset; // this frame's curve BVH node words
            this.curveBvhOffset = (bvh.nodes.length + bvh.tris.length) / 4;
            this.curvePrimOffset = this.curveBvhOffset + nodeWords;
        }
        // Setblob in place into the worst-case-sized buffer (allocated in the ctor).
        this.buffers["bvhNodes"]!.setBlob(bvhMerged.subarray(0, Math.min(bvhMerged.length, this.bvhCapacityBytes / 4)));
        return true;
    }

    /** Uploads last frame's deformed positions to prevVertices, then remembers
     *  the current ones for the next frame (first frame keeps the bind pose). */
    private rollPrevVertices(meshID: number, vbOffset: number, deformed: StaticVertex[]): void {
        const prevBuf = this.buffers["prevVertices"];
        if (!prevBuf) return;
        const pending = this.pendingPrevVerts.get(meshID);
        if (pending) prevBuf.setBlob(pending, vbOffset * 16);
        this.pendingPrevVerts.set(meshID, packPrevVertices(deformed));
    }

    /**
     * Mirrors Animatable::updateFromAnimation for the camera and analytic lights:
     * re-poses each from its bound node's current global matrix. glTF cameras and
     * lights look down local -Z with local +Y up; point-light position is the node
     * origin. Area lights (Rect/Disc/Sphere) ride the full node transform.
     */
    private updateAnimatedCameraAndLights(globals: float4x4[]): void {
        const ZERO = new float3(0, 0, 0);
        const FWD = new float3(0, 0, -1);
        const UP = new float3(0, 1, 0);
        const animatedCamera = this.cameraList[this.animatedCameraIndex];
        if (this.cameraNodeID !== undefined && globals[this.cameraNodeID] && animatedCamera?.animated !== false) {
            const g = globals[this.cameraNodeID]!;
            const pos = transformPoint(g, ZERO);
            const fwd = normalize3(transformVector(g, FWD));
            const camera = animatedCamera!;
            camera.setPosition(pos);
            camera.setTarget(new float3(pos.x + fwd.x, pos.y + fwd.y, pos.z + fwd.z));
            camera.setUpVector(normalize3(transformVector(g, UP)));
        }
        let lightsDirty = false;
        for (const light of this.analyticLights) {
            if (light.nodeID === undefined || !globals[light.nodeID] || (light as SceneLight).animated === false) continue;
            const g = globals[light.nodeID]!;
            const isArea = light.type === LightType.Rect || light.type === LightType.Disc || light.type === LightType.Sphere;
            if (isArea) {
                light.transMat = g;
            } else {
                light.posW = transformPoint(g, ZERO);
                light.dirW = normalize3(transformVector(g, FWD));
            }
            lightsDirty = true;
        }
        if (lightsDirty) this.updateLights();
    }

    /** Per-emissive-triangle flux in LightCollection order (for power sampling). */
    getEmissiveFluxes(): Float32Array {
        return this.emissiveFluxes;
    }

    /** Emissive triangles as LightBVH builder inputs (LightCollection order). */
    getEmissiveTriangles(): EmissiveTriangleInput[] {
        return this.emissiveTriangles;
    }

    getEnvMap(): EnvMap | null {
        return this.envMap;
    }

    /** Mirrors Scene::hasGeometryType(Curve). */
    get hasCurves(): boolean {
        return this.curveDescs.length > 0;
    }

    /** Mirrors Scene::hasGeometryType. */
    hasGeometryType(type: GeometryType): boolean {
        switch (type) {
            case GeometryType.TriangleMesh:
                return this.triangleTotal > 0;
            case GeometryType.DisplacedTriangleMesh:
                return this.hasDisplaced;
            case GeometryType.Curve:
                return this.curveDescs.length > 0;
            case GeometryType.SDFGrid:
                return this.sdfGrids.length > 0;
            case GeometryType.Custom:
                return this.customPrimitives.length > 0;
            default:
                return false;
        }
    }

    // ---- Custom primitives (Scene::addCustomPrimitive and friends) ----
    //
    // Like native, these are user IDs plus AABBs for passes that bring their
    // own intersection code; they never enter the triangle BVH, so the shipped
    // passes (which have no intersection shader for them) pass straight through.

    private customPrimitives: { userID: number; aabb: { min: [number, number, number]; max: [number, number, number] } }[] = [];

    /** Mirrors Scene::getCustomPrimitiveCount. */
    getCustomPrimitiveCount(): number {
        return this.customPrimitives.length;
    }

    /** Mirrors Scene::getCustomPrimitive: each primitive has exactly one AABB, at its own index. */
    getCustomPrimitive(index: number): { userID: number; aabbOffset: number } {
        const p = this.customPrimitives[index];
        if (!p) throw new RuntimeError(`Scene.getCustomPrimitive: 'index' (${index}) is out of range`);
        return { userID: p.userID, aabbOffset: index };
    }

    /** Mirrors Scene::getCustomPrimitiveAABB. */
    getCustomPrimitiveAABB(index: number): { min: [number, number, number]; max: [number, number, number] } {
        const p = this.customPrimitives[index];
        if (!p) throw new RuntimeError(`Scene.getCustomPrimitiveAABB: 'index' (${index}) is out of range`);
        return { min: [...p.aabb.min], max: [...p.aabb.max] };
    }

    /** Mirrors Scene::addCustomPrimitive; returns the new primitive's index. */
    addCustomPrimitive(userID: number, aabb: { min: [number, number, number]; max: [number, number, number] }): number {
        this.customPrimitives.push({ userID, aabb: { min: [...aabb.min], max: [...aabb.max] } });
        return this.customPrimitives.length - 1;
    }

    /** Mirrors Scene::removeCustomPrimitives (half-open [first, last)). */
    removeCustomPrimitives(first: number, last: number): void {
        if (!(first >= 0 && first <= last && last <= this.customPrimitives.length)) {
            throw new RuntimeError(`Scene.removeCustomPrimitives: invalid range [${first}, ${last})`);
        }
        this.customPrimitives.splice(first, last - first);
    }

    /** Mirrors Scene::updateCustomPrimitive. */
    updateCustomPrimitive(index: number, aabb: { min: [number, number, number]; max: [number, number, number] }): void {
        const p = this.customPrimitives[index];
        if (!p) throw new RuntimeError(`Scene.updateCustomPrimitive: 'index' (${index}) is out of range`);
        p.aabb = { min: [...aabb.min], max: [...aabb.max] };
    }

    /**
     * Mirrors Scene::RenderSettings (python `scene.renderSettings.useEnvLight = False`,
     * Mogwai "Render Settings"): master switches ANDed with resource presence below.
     * The RenderGraph recompiles passes when these change (native RenderSettingsChanged).
     */
    private _renderSettings = { useEnvLight: true, useAnalyticLights: true, useEmissiveLights: true, useGridVolumes: true, diffuseAlbedoMultiplier: 1 };
    get renderSettings(): { useEnvLight: boolean; useAnalyticLights: boolean; useEmissiveLights: boolean; useGridVolumes: boolean; diffuseAlbedoMultiplier: number } {
        return this._renderSettings;
    }
    /** Mirrors Scene::setRenderSettings (python `scene.renderSettings = SceneRenderSettings(...)`). */
    set renderSettings(v: Partial<{ useEnvLight: boolean; useAnalyticLights: boolean; useEmissiveLights: boolean; useGridVolumes: boolean; diffuseAlbedoMultiplier: number }>) {
        const r = this._renderSettings;
        for (const k of Object.keys(r) as (keyof typeof r)[]) {
            const x = (v as Record<string, unknown>)[k];
            if (x !== undefined) (r as Record<string, unknown>)[k] = k === "diffuseAlbedoMultiplier" ? Number(x) : Boolean(x);
        }
    }
    setRenderSettings(v: Partial<Scene["renderSettings"]>): void {
        this.renderSettings = v;
    }

    /** Snapshot key for change detection (native compares mRenderSettings != mPrevRenderSettings). */
    getRenderSettingsKey(): string {
        const r = this.renderSettings;
        return `${+r.useEnvLight}${+r.useAnalyticLights}${+r.useEmissiveLights}${+r.useGridVolumes}|${r.diffuseAlbedoMultiplier}`;
    }

    /** Mirrors Scene::useAnalyticLights(). */
    get useAnalyticLights(): boolean {
        return this.renderSettings.useAnalyticLights && this.lightCount > 0;
    }

    /** Mirrors Scene::useGridVolumes(). */
    get useGridVolumes(): boolean {
        return this.renderSettings.useGridVolumes && this.gridCount > 0;
    }

    /** Mirrors Scene::useEmissiveLights(). v1 checks material flags; the
     *  LightCollection active-triangle count refines this when NEE lands. */
    get useEmissiveLights(): boolean {
        return this.renderSettings.useEmissiveLights && this.hasEmissiveMaterials;
    }

    /** Mirrors Scene::useEnvBackground(). */
    get useEnvBackground(): boolean {
        return this.envMap !== null;
    }

    /** Mirrors Scene::useEnvLight(). */
    get useEnvLight(): boolean {
        return this.renderSettings.useEnvLight && this.envMap !== null && this.envMap.intensity > 0;
    }

    /** Mirrors Scene::getSceneDefines(). */
    /** Binds gScene.sdfGrid0 for the NormalizedDenseGrid implementation. */
    private bindSdfNd(scene: ShaderVar, grid: NDSDFGrid): void {
        if (!this.sdfAtlasTexture) {
            const atlas = grid.buildAtlas();
            this.sdfAtlasTexture = new Texture(this.device, {
                type: ResourceType.Texture3D,
                width: atlas.width,
                height: atlas.height,
                depth: atlas.depth,
                format: ResourceFormat.R8Snorm,
                bindFlags: ResourceBindFlags.ShaderResource,
                name: "Scene::sdfGrid0Atlas",
            });
            this.sdfAtlasTexture.setSubresourceBlob(0, 0, new Uint8Array(atlas.data.buffer));
            // Native NDSDFGrid::SharedData sampler: linear min/mag/mip, clamp.
            this.sdfSampler = new Sampler(this.device, {
                magFilter: TextureFilteringMode.Linear,
                minFilter: TextureFilteringMode.Linear,
                mipFilter: TextureFilteringMode.Linear,
                addressModeU: TextureAddressingMode.Clamp,
                addressModeV: TextureAddressingMode.Clamp,
                addressModeW: TextureAddressingMode.Clamp,
            });
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["atlasTexture"] = this.sdfAtlasTexture;
            v["sampler"] = this.sdfSampler!;
            v["lodCount"] = grid.lodCount;
            v["coarsestLODAsLevel"] = grid.coarsestLODAsLevel;
            v["coarsestLODGridWidth"] = grid.coarsestLODGridWidth;
            v["coarsestLODNormalizationFactor"] = grid.coarsestLODNormalizationFactor;
            v["narrowBandThickness"] = grid.narrowBandThickness;
        } catch (e) {
            console.error(`# sdfGrid0 (ND) bind failed: ${e}`);
        }
    }

    /**
     * One merged float4 buffer (16-storage-buffer budget): a 2-float4 header per SDF instance
     * (BVH root node, and for SBS its grid's virtualGridWidth, virtualBricksPerAxis, indirection
     * z offset, normalizationFactor), then every distinct grid's BVH nodes (2 float4/node), then
     * prim indices packed 4-per-float4.
     */
    private buildSdfBvhBuffer(sbs: { grids: SDFSBS[]; packed: PackedSBS } | null, aabbs: { min: [number, number, number]; max: [number, number, number] }[]): { buf: Buffer; primOffset: number } {
        const gridsOf = sbs ? sbs.grids : [this.sdfGrids[0]!.grid];
        const ranges = sbs ? sbs.grids.map((g, i) => [sbs.packed.brickOffsets[i]!, g.brickCount] as const) : [[0, aabbs.length] as const];
        const bvhs = ranges.map(([first, count]) => buildAabbBvh(aabbs.slice(first, first + count)));
        const header = this.sdfGrids.length * 2;
        const nodeTotal = bvhs.reduce((n, b) => n + b.nodeCount, 0);
        const primTotal = bvhs.reduce((n, b) => n + b.primIndices.length, 0);
        const primOffset = header + nodeTotal * 2;
        const merged = new Float32Array((primOffset + Math.ceil(primTotal / 4)) * 4);
        const u32 = new Uint32Array(merged.buffer);
        const roots: number[] = [];
        let [nodeBase, primBase] = [0, 0];
        bvhs.forEach((bvh, g) => {
            roots.push(nodeBase);
            const nodes = new Uint32Array(bvh.nodes.buffer, bvh.nodes.byteOffset, bvh.nodeCount * 8);
            const at = (header + nodeBase * 2) * 4;
            u32.set(nodes, at);
            for (let n = 0; n < bvh.nodeCount; n++) {
                // Leaves index the prim list, interior nodes their left child.
                u32[at + n * 8 + 3]! += u32[at + n * 8 + 7]! > 0 ? primBase : header / 2 + nodeBase;
            }
            const first = ranges[g]![0];
            for (let i = 0; i < bvh.primIndices.length; i++) u32[primOffset * 4 + primBase + i] = bvh.primIndices[i]! + first;
            nodeBase += bvh.nodeCount;
            primBase += bvh.primIndices.length;
        });
        this.sdfGrids.forEach((d, i) => {
            const g = gridsOf.indexOf(d.grid);
            u32[i * 8] = header / 2 + roots[g]!;
            if (sbs) {
                const grid = sbs.grids[g]!;
                u32.set([grid.gridWidth, grid.virtualBricksPerAxis, sbs.packed.zOffsets[g]!], i * 8 + 1);
                merged[i * 8 + 4] = grid.normalizationFactor;
            }
        });
        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const buf = new Buffer(this.device, { size: merged.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::sdfBvh" });
        buf.setBlob(new Uint8Array(merged.buffer));
        return { buf, primOffset };
    }

    /** Distinct SBS grids of the scene, packed (see packSBSGrids). */
    private getPackedSBS(): { grids: SDFSBS[]; packed: PackedSBS } {
        if (!this.sbsPacked) {
            const grids = [...new Set(this.sdfGrids.map((d) => d.grid))] as SDFSBS[];
            this.sbsPacked = { grids, packed: packSBSGrids(grids) };
        }
        return this.sbsPacked;
    }

    /** Binds gScene.sdfGrid0 for the SparseBrickSet implementation (all grids packed; per-grid
     *  fields are patched per instance by the Scene.slang override's getSDFGrid). */
    private bindSdfSbs(scene: ShaderVar, grid: SDFSBS): void {
        const { packed } = this.getPackedSBS();
        if (!this.sbsResources) {
            const storage = ResourceBindFlags.ShaderResource;
            // AABB StructuredBuffer: 32-byte stride (min.xyz @0, max.xyz @16).
            const aabbData = new Float32Array(packed.aabbs.length * 8);
            packed.aabbs.forEach((a, i) => {
                aabbData.set(a.min, i * 8);
                aabbData.set(a.max, i * 8 + 4);
            });
            const aabbs = new Buffer(this.device, {
                size: Math.max(aabbData.byteLength, 32),
                structSize: 32,
                bindFlags: storage | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "Scene::sdfGrid0Aabbs",
            });
            aabbs.setBlob(new Uint8Array(aabbData.buffer));

            const [iw, ih, id] = packed.indirectionDims;
            const indirection = new Texture(this.device, {
                type: ResourceType.Texture3D,
                width: iw,
                height: ih,
                depth: id,
                format: ResourceFormat.R32Uint,
                bindFlags: storage,
                name: "Scene::sdfGrid0Indirection",
            });
            indirection.setSubresourceBlob(0, 0, new Uint8Array(packed.indirection.buffer));

            // Compressed SBS: a BC4Snorm texture, as natively (SDFSBS::mCompressed).
            const [bw, bh] = packed.brickTextureDimensions;
            const bricks = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: bw,
                height: bh,
                format: packed.compressed ? ResourceFormat.BC4Snorm : ResourceFormat.R32Float,
                bindFlags: packed.compressed ? ResourceBindFlags.ShaderResource : storage,
                name: "Scene::sdfGrid0Bricks",
            });
            bricks.setSubresourceBlob(0, 0, packed.compressed ? encodeBC4Texture(packed.brickTexture, bw, bh) : new Uint8Array(packed.brickTexture.buffer));
            // Native SDFSBS::SharedData sampler: linear, clamp (brick edges).
            const sampler = new Sampler(this.device, {
                magFilter: TextureFilteringMode.Linear,
                minFilter: TextureFilteringMode.Linear,
                mipFilter: TextureFilteringMode.Linear,
                addressModeU: TextureAddressingMode.Clamp,
                addressModeV: TextureAddressingMode.Clamp,
                addressModeW: TextureAddressingMode.Clamp,
            });
            this.sbsResources = { aabbs, indirection, bricks, sampler };
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["aabbs"] = this.sbsResources.aabbs;
            v["indirectionBuffer"] = this.sbsResources.indirection;
            v["bricks"] = this.sbsResources.bricks;
            v["sampler"] = this.sbsResources.sampler;
            v["virtualGridWidth"] = grid.gridWidth;
            v["virtualBricksPerAxis"] = grid.virtualBricksPerAxis;
            v["bricksPerAxis"] = packed.bricksPerAxis;
            v["brickTextureDimensions"] = packed.brickTextureDimensions;
            v["brickWidth"] = grid.brickWidth;
            v["normalizationFactor"] = grid.normalizationFactor;
            v["indirectionZOffset"] = 0;
        } catch (e) {
            console.error(`# sdfGrid0 (SBS) bind failed: ${e}`);
        }
    }

    /** Binds gScene.sdfGrid0 for the SparseVoxelSet implementation. */
    private bindSdfSvs(scene: ShaderVar, grid: SDFSVS): void {
        if (!this.svsResources) {
            const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            // WebFalcorSVSVoxel StructuredBuffer: AABB (32B) + SDFSVSVoxel (80B)
            // merged, 112-byte stride (the SVS.slang override reads .aabb/.voxel).
            const n = Math.max(grid.voxelCount, 1);
            const combined = new Float32Array(n * 28); // 112 bytes = 28 words
            const combinedU = new Uint32Array(combined.buffer);
            for (let i = 0; i < grid.voxelCount; i++) {
                const a = grid.aabbs[i]!;
                combined.set(a.min, i * 28); // aabb.min @0
                combined.set(a.max, i * 28 + 4); // aabb.max @16
                combinedU.set(grid.voxelData.subarray(i * 20, i * 20 + 20), i * 28 + 8); // voxel @32
            }
            const voxels = new Buffer(this.device, {
                size: combined.byteLength,
                structSize: 112,
                bindFlags: storage,
                memoryType: MemoryType.DeviceLocal,
                name: "Scene::sdfGrid0Voxels",
            });
            voxels.setBlob(new Uint8Array(combined.buffer));
            this.svsResources = { voxels };
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["voxels"] = this.svsResources.voxels;
            v["virtualGridWidth"] = grid.gridWidth;
            v["oneDivVirtualGridWidth"] = 1 / grid.gridWidth;
            v["normalizationFactor"] = grid.normalizationFactor;
        } catch (e) {
            console.error(`# sdfGrid0 (SVS) bind failed: ${e}`);
        }
    }

    /** Binds gScene.sdfGrid0 for the SparseVoxelOctree implementation. */
    private bindSdfSvo(scene: ShaderVar, grid: SDFSVO): void {
        if (!this.svoResources) {
            const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            // SDFSVOVoxel StructuredBuffer: 24-byte stride (relationData + pad +
            // uint2 locationCode + uint2 packedValues).
            const svo = new Buffer(this.device, {
                size: Math.max(grid.svoData.byteLength, 24),
                structSize: 24,
                bindFlags: storage,
                memoryType: MemoryType.DeviceLocal,
                name: "Scene::sdfGrid0Svo",
            });
            svo.setBlob(new Uint8Array(grid.svoData.buffer, grid.svoData.byteOffset, grid.svoData.byteLength));
            this.svoResources = { svo };
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["svo"] = this.svoResources.svo;
            v["levelCount"] = grid.levelCount;
        } catch (e) {
            console.error(`# sdfGrid0 (SVO) bind failed: ${e}`);
        }
    }

    getSceneDefines(): DefineList {
        return new DefineList().addAll({
            SCENE_GEOMETRY_TYPES:
                (1 << GeometryType.TriangleMesh) |
                (this.sdfGrids.length > 0 ? 1 << GeometryType.SDFGrid : 0) |
                (this.curveDescs.length > 0 ? 1 << GeometryType.Curve : 0) |
                (this.hasDisplaced ? 1 << GeometryType.DisplacedTriangleMesh : 0),
            SCENE_GRID_COUNT: this.gridCount,
            // Mirrors Scene::getSceneSDFGridDefines (defaults for all types:
            // VoxelSphereTracing, NumericDiscontinuous, 256 iterations).
            SCENE_SDF_GRID_COUNT: this.sdfGrids.length,
            SCENE_SDF_GRID_MAX_LOD_COUNT: this.sdfGrids.length > 0 ? Math.max(...this.sdfGrids.map((g) => (g.grid instanceof SDFSBS || g.grid instanceof SDFSVS ? 32 - Math.clz32(g.grid.gridWidth) : g.grid instanceof SDFSVO ? g.grid.levelCount : g.grid.lodCount))) : 0,
            // 1 = NormalizedDenseGrid, 2 = SparseVoxelSet, 3 = SparseBrickSet (all grids in a scene share a type).
            SCENE_SDF_GRID_IMPLEMENTATION: this.sdfGrids.length > 0 ? (this.sdfGrids[0]!.grid instanceof SDFSBS ? 3 : this.sdfGrids[0]!.grid instanceof SDFSVS ? 2 : this.sdfGrids[0]!.grid instanceof SDFSVO ? 4 : 1) : 0,
            SCENE_SDF_GRID_IMPLEMENTATION_NDSDF: 1,
            SCENE_SDF_GRID_IMPLEMENTATION_SVS: 2,
            SCENE_SDF_GRID_IMPLEMENTATION_SBS: 3,
            SCENE_SDF_GRID_IMPLEMENTATION_SVO: 4,
            SCENE_SDF_NO_INTERSECTION_METHOD: 0,
            SCENE_SDF_NO_VOXEL_SOLVER: 1,
            SCENE_SDF_VOXEL_SPHERE_TRACING: 2,
            SCENE_SDF_NO_GRADIENT_EVALUATION_METHOD: 0,
            SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS: 1,
            SCENE_SDF_GRADIENT_NUMERIC_CONTINUOUS: 2,
            SCENE_SDF_VOXEL_INTERSECTION_METHOD: 2,
            SCENE_SDF_GRADIENT_EVALUATION_METHOD: 1,
            SCENE_SDF_SOLVER_MAX_ITERATION_COUNT: 256,
            SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS: 1,
            SCENE_HAS_INDEXED_VERTICES: 1,
            SCENE_HAS_16BIT_INDICES: 0,
            SCENE_HAS_32BIT_INDICES: 1,
            SCENE_INDEX_BUFFER_COUNT: 1,
            SCENE_INDEX_BUFFER_INDEX_BITS: 1,
            SCENE_VERTEX_BUFFER_COUNT: 1,
            SCENE_VERTEX_BUFFER_INDEX_BITS: 1,
            HIT_INFO_DEFINES: 1,
            HIT_INFO_USE_COMPRESSION: 0,
            ...this.hitInfoDefines(),
            MATERIAL_SYSTEM_SAMPLER_DESC_COUNT: 16,
            MATERIAL_SYSTEM_TEXTURE_DESC_COUNT: this.textureCount,
            MATERIAL_SYSTEM_BUFFER_DESC_COUNT: 1,
            MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT: 1,
            MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED: 0,
            MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS: this.materialDescs.some(
                (m) => (m.header?.materialType ?? MaterialType.Standard) === MaterialType.Standard && m.basic.shadingModel === ShadingModel.SpecGloss,
            )
                ? 1
                : 0,
            MATERIAL_SYSTEM_USE_LIGHT_PROFILE: this.lightProfile ? 1 : 0,
            FALCOR_MATERIAL_INSTANCE_SIZE: 256,
            // Static material dispatch (MaterialFactory override) — mirrors
            // MaterialSystem::getTypeConformances() type registration.
            // Material-less scenes (pure volumes) still need one registered type:
            // the factory's fallback return must exist for WGSL (E41009).
            WEBFALCOR_MTL_STANDARD: this.materialTypes.has(MaterialType.Standard) || this.materialTypes.size === 0 ? 1 : 0,
            WEBFALCOR_MTL_CLOTH: this.materialTypes.has(MaterialType.Cloth) ? 1 : 0,
            WEBFALCOR_MTL_MERL: this.materialTypes.has(MaterialType.MERL) ? 1 : 0,
            WEBFALCOR_MTL_MERLMIX: this.materialTypes.has(MaterialType.MERLMix) ? 1 : 0,
            WEBFALCOR_MTL_RGL: this.materialTypes.has(MaterialType.RGL) ? 1 : 0,
            WEBFALCOR_MTL_HAIR: this.materialTypes.has(MaterialType.Hair) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIFFUSE: this.materialTypes.has(MaterialType.PBRTDiffuse) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIFFUSE_TRANSMISSION: this.materialTypes.has(MaterialType.PBRTDiffuseTransmission) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_CONDUCTOR: this.materialTypes.has(MaterialType.PBRTConductor) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIELECTRIC: this.materialTypes.has(MaterialType.PBRTDielectric) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_COATED_CONDUCTOR: this.materialTypes.has(MaterialType.PBRTCoatedConductor) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_COATED_DIFFUSE: this.materialTypes.has(MaterialType.PBRTCoatedDiffuse) ? 1 : 0,
            // std::to_string(float), as natively.
            SCENE_DIFFUSE_ALBEDO_MULTIPLIER: this.renderSettings.diffuseAlbedoMultiplier.toFixed(6),
            FALCOR_NVAPI_AVAILABLE: 0,
            SAMPLE_GENERATOR_TYPE: 0, // TinyUniform (SampleGeneratorType.slangh)
        });
    }

    /**
     * Uploads grid-volume GPU data after resolve() populates gridVolumes
     * (web divergence: volumes load asynchronously after construction).
     * WGSL has no binding arrays: all grids share one NanoVDB buffer (gScene.gridData, each at a
     * 32-byte aligned offset) described by gScene.gridInfos.
     */
    finalizeGridVolumes(): void {
        if (this.gridVolumes.length === 0) return;
        const grids: import("./Volume/Grid.js").Grid[] = [];
        const gridIndex = (g: import("./Volume/Grid.js").Grid | undefined): number => {
            if (!g) return 0xffffffff;
            let i = grids.indexOf(g);
            if (i < 0) {
                i = grids.length;
                grids.push(g);
            }
            return i;
        };

        // GridVolumeData: 192 B per volume (2x float4x4 + 4x 16 B rows).
        const data = new ArrayBuffer(this.gridVolumes.length * 192);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        this.gridVolumes.forEach((vol, vi) => {
            const o = vi * 48; // floats
            // Mirrors Scene::updateGridVolumes: the uploaded transform merges the
            // volume transform (identity: no animation yet) with the density
            // grid's index->world map, so invTransform maps world -> INDEX space.
            const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
            f32.set(vol.densityGrid ? vol.densityGrid.indexToWorldMatrix : identity, o);
            f32.set(vol.densityGrid ? vol.densityGrid.worldToIndexMatrix : identity, o + 16);
            const b = vol.bounds ?? { min: [0, 0, 0], max: [0, 0, 0] };
            f32.set(b.min, o + 32);
            f32[o + 35] = vol.densityScale;
            f32.set(b.max, o + 36);
            f32[o + 39] = vol.emissionScale;
            u32[o + 40] = gridIndex(vol.densityGrid);
            u32[o + 41] = gridIndex(vol.emissionGrid);
            u32[o + 42] = 0; // flags (emission mode Direct)
            f32[o + 43] = vol.anisotropy;
            f32[o + 44] = vol.albedo.x;
            f32[o + 45] = vol.albedo.y;
            f32[o + 46] = vol.albedo.z;
            f32[o + 47] = vol.emissionTemperature;
        });
        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const volBuf = new Buffer(this.device, { size: data.byteLength, structSize: 192, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::gridVolumes" });
        volBuf.setBlob(new Uint8Array(data));
        this.buffers["gridVolumesData"] = volBuf;

        if (grids.length > 0) {
            const offsets: number[] = [];
            let size = 0;
            for (const g of grids) {
                offsets.push(size);
                size = Math.ceil((size + g.gridBuffer.byteLength) / 32) * 32;
            }
            const gridData = new Uint8Array(size);
            // GridInfo: 48 B (minIndex, minValue, maxIndex, maxValue, baseAddress, pad).
            const info = new ArrayBuffer(grids.length * 48);
            const infoF = new Float32Array(info);
            const infoI = new Int32Array(info);
            grids.forEach((g, i) => {
                gridData.set(new Uint8Array(g.gridBuffer.buffer, g.gridBuffer.byteOffset, g.gridBuffer.byteLength), offsets[i]!);
                infoI.set(g.minIndex, i * 12);
                infoF[i * 12 + 3] = g.minValue;
                infoI.set(g.maxIndex, i * 12 + 4);
                infoF[i * 12 + 7] = g.maxValue;
                infoI[i * 12 + 8] = offsets[i]!;
            });
            const dataBuf = new Buffer(this.device, { size, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::gridData" });
            dataBuf.setBlob(gridData);
            this.buffers["gridData"] = dataBuf;
            const infoBuf = new Buffer(this.device, { size: info.byteLength, structSize: 48, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::gridInfos" });
            infoBuf.setBlob(new Uint8Array(info));
            this.buffers["gridInfos"] = infoBuf;
            const g = grids[0]!;
            this.grid0Stats = { minIndex: g.minIndex, minValue: g.minValue, maxIndex: g.maxIndex, maxValue: g.maxValue };
        }
        this.gridCount = grids.length;
    }

    /**
     * Mirrors Scene::updateSDFGrids: re-bakes any grid whose primitive list was
     * edited and drops the GPU resources built from it, so the next bind
     * rebuilds them. Returns true if anything changed (reset accumulation).
     */
    updateSDFGrids(): boolean {
        let changed = false;
        for (const desc of this.sdfGrids) changed = (desc.grid.primitives?.rebuild() ?? false) || changed;
        if (changed) this.invalidateSDFResources();
        return changed;
    }

    /** Drops the lazily built SDF GPU resources; bindShaderData recreates them.
     *  Old objects are replaced, not destroyed, so in-flight submits stay valid. */
    private invalidateSDFResources(): void {
        this.sdfAtlasTexture = null;
        this.sbsResources = null;
        this.sbsPacked = null;
        this.svsResources = null;
        this.svoResources = null;
        this.sdfBvhBuffers = null;
    }

    /**
     * Mirrors Scene::updateGridVolumes' playback step: advances every volume's
     * grid sequence to `timeSec` and rebuilds the grid bindings when a frame
     * changed. Returns true if anything moved (the caller can reset accumulation).
     */
    updateGridVolumePlayback(timeSec: number): boolean {
        let changed = false;
        for (const volume of this.gridVolumes) changed = volume.updatePlayback(timeSec) || changed;
        if (changed) this.finalizeGridVolumes();
        return changed;
    }

    /** Mirrors Scene::bindShaderData: fills the gScene parameter block. */
    bindShaderData(root: ShaderVar): void {
        if (this.lightsDirty) this.updateLights();
        const scene = root["gScene"];

        // Camera (uniforms in the block's default buffer; statically-unused sets no-op).
        const cam = this.camera.getData();
        const c = scene["camera"]["data"];
        c["viewMat"] = cam.viewMat;
        c["projMat"] = cam.projMat;
        c["viewProjMat"] = cam.viewProjMat;
        c["viewProjMatNoJitter"] = cam.viewProjMatNoJitter;
        c["prevViewProjMatNoJitter"] = cam.prevViewProjMatNoJitter;
        c["invViewProj"] = cam.invViewProj;
        c["posW"] = cam.posW.toArray();
        c["focalLength"] = cam.focalLength;
        c["up"] = cam.up.toArray();
        c["aspectRatio"] = cam.aspectRatio;
        c["target"] = cam.target.toArray();
        c["nearZ"] = cam.nearZ;
        c["cameraU"] = cam.cameraU.toArray();
        c["farZ"] = cam.farZ;
        c["cameraV"] = cam.cameraV.toArray();
        c["jitterX"] = cam.jitterX;
        c["cameraW"] = cam.cameraW.toArray();
        c["jitterY"] = cam.jitterY;
        c["focalDistance"] = cam.focalDistance;
        c["apertureRadius"] = cam.apertureRadius;
        c["shutterSpeed"] = cam.shutterSpeed;
        c["ISOSpeed"] = cam.ISOSpeed;

        // Geometry.
        scene["worldMatrices"] = this.buffers["worldMatrices"]!;
        scene["prevWorldMatrices"] = this.buffers["prevWorldMatrices"] ?? this.buffers["worldMatrices"]!;
        scene["webfalcorInvTransposeOffset"] = this.invTransposeOffset;
        scene["geometryInstances"] = this.buffers["geometryInstances"]!;
        scene["meshes"] = this.buffers["meshes"]!;
        scene["vertices"]["data0"] = this.buffers["vertices"]!;
        scene["webfalcorBvhNodes"] = this.buffers["bvhNodes"]!;
        scene["webfalcorBvhTrisOffset"] = this.bvhTrisOffset;
        scene["lights"] = this.buffers["lights"]!;
        scene["lightCount"] = this.lightCount;
        scene["prevVertices"] = this.buffers["prevVertices"] ?? this.buffers["vertices"]!;
        // Curve buffers (dummies keep DCE survivors bound in curve-less scenes).
        scene["curveVertices"] = this.buffers["curveVertices"] ?? this.buffers["curveDummy"]!;
        scene["prevCurveVertices"] = this.buffers["curveVertices"] ?? this.buffers["curveDummy"]!;
        scene["curveIndices"] = this.buffers["curveIndices"] ?? this.buffers["curveDummy"]!;
        if (this.buffers["curves"]) scene["curves"] = this.buffers["curves"]!;
        try {
            // Only referenced by volume-aware passes (binding absent otherwise).
            scene["gridVolumeCount"] = this.gridVolumes.length;
            scene["gridVolumes"] = this.buffers["gridVolumesData"] ?? this.buffers["gridVolumeDummy"]!;
        } catch (e) {
            if (this.gridVolumes.length > 0) console.error(`# gridVolumes bind failed: ${e}`);
        }
        scene["indexData"]["data0"] = this.buffers["indices"]!;

        // SDF grids (one instance set; sdfGrid0 bindings survive only with
        // SCENE_SDF_GRID_COUNT > 0 — trySet semantics via try/catch).
        const grid0 = this.sdfGrids.length > 0 ? this.sdfGrids[0]!.grid : null;
        const sbsGrid = grid0 instanceof SDFSBS ? grid0 : null;
        const svsGrid = grid0 instanceof SDFSVS ? grid0 : null;
        const svoGrid = grid0 instanceof SDFSVO ? grid0 : null;
        // SBS/SVS traverse a BVH over their primitive AABBs (bricks/voxels).
        const distinctGrids = new Set(this.sdfGrids.map((d) => d.grid)).size;
        if (distinctGrids > 1 && !sbsGrid) throw new RuntimeError("Scene: several distinct SDF grids are supported for SBS only (gScene.sdfGrid0; WGSL has no binding arrays)");
        const sdfAabbs = sbsGrid ? this.getPackedSBS().packed.aabbs : svsGrid ? svsGrid.aabbs : null;
        try {
            scene["webfalcorSdfInstanceFirst"] = this.sdfInstanceFirst;
            scene["webfalcorSdfInstanceCount"] = this.sdfGrids.length;
            scene["webfalcorSdfBrickCount"] = sdfAabbs ? sdfAabbs.length : 0;
        } catch {
            /* SDF-less kernel variant */
        }
        try {
            scene["webfalcorCurveInstanceFirst"] = this.curveInstanceFirst;
            scene["webfalcorCurveInstanceCount"] = this.curveDescs.length;
            scene["webfalcorCurveBvhOffset"] = this.curveBvhOffset;
            scene["webfalcorCurvePrimOffset"] = this.curvePrimOffset;
        } catch {
            /* curve-less kernel variant */
        }
        try {
            scene["webfalcorDisplacedBvhOffset"] = this.displacedBvhOffset;
            scene["webfalcorDisplacedPrimOffset"] = this.displacedPrimOffset;
        } catch {
            /* curve-less kernel variant */
        }
        if (sdfAabbs) {
            if (!this.sdfBvhBuffers) this.sdfBvhBuffers = this.buildSdfBvhBuffer(sbsGrid ? this.getPackedSBS() : null, sdfAabbs);
            try {
                scene["webfalcorSdfBvh"] = this.sdfBvhBuffers.buf;
                scene["webfalcorSdfPrimOffset"] = this.sdfBvhBuffers.primOffset;
            } catch {
                /* bindings absent in NDSDF/SVO kernel variant */
            }
        }
        if (this.sdfGrids.length > 0) {
            if (sbsGrid) this.bindSdfSbs(scene, sbsGrid);
            else if (svsGrid) this.bindSdfSvs(scene, svsGrid);
            else if (svoGrid) this.bindSdfSvo(scene, svoGrid);
            else this.bindSdfNd(scene, this.sdfGrids[0]!.grid as NDSDFGrid);
        }

        // Env map (dummy black texture + zeroed uniforms when absent).
        if (this.envMap) {
            this.envMap.bindShaderData(scene["envMap"] as ShaderVar);
        } else {
            scene["envMap"]["envMap"] = this.dummyTexture;
            scene["envMap"]["envSampler"] = this.sampler;
        }

        // Emissive geometry (LightCollection.slang).
        const lightCollection = scene["lightCollection"];
        lightCollection["triangleCount"] = this.emissiveTriangleCount;
        lightCollection["activeTriangleCount"] = this.emissiveTriangleCount;
        lightCollection["meshCount"] = this.emissiveMeshCount;
        lightCollection["triangleData"] = this.buffers["emissiveTriangles"]!;
        lightCollection["activeTriangles"] = this.buffers["emissiveActiveTriangles"]!;
        lightCollection["triToActiveMapping"] = this.buffers["emissiveTriToActive"]!;
        lightCollection["fluxData"] = this.buffers["emissiveFlux"]!;
        lightCollection["meshData"] = this.buffers["emissiveMeshData"]!;
        lightCollection["perMeshInstanceOffset"] = this.buffers["emissivePerMeshInstanceOffset"]!;

        // Grids: the shared NanoVDB buffer and per-grid infos (dummies keep SCENE_GRID_COUNT=0
        // variants bindable). Bricked-grid textures stay dummies: the upstream consumers use the
        // NanoVDB lookup path.
        scene["gridData"] = this.buffers["gridData"] ?? this.buffers["materialBuffer0"]!;
        this.buffers["gridInfos"] ??= new Buffer(this.device, { size: 48, structSize: 48, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "Scene::gridInfos(empty)" });
        scene["gridInfos"] = this.buffers["gridInfos"];
        scene["gridRangeTex"] = this.gridRangeTex;
        scene["gridIndirectionTex"] = this.gridIndirectionTex;
        scene["gridAtlasTex"] = this.gridAtlasTex;

        // Light profile (disabled; dummy bindings).
        if (this.lightProfile) {
            this.lightProfile.bindShaderData(scene["materials"]["lightProfile"] as ShaderVar);
        } else {
            scene["materials"]["lightProfile"]["texture"] = this.dummyTexture;
            scene["materials"]["lightProfile"]["sampler"] = this.sampler;
        }

        // Material system.
        const materials = scene["materials"];
        materials["materialCount"] = this.materialCount;
        materials["materialData"] = this.buffers["materialData"]!;
        materials["materialSampler0"] = this.sampler;
        for (let i = 0; i < kMaxTextureBuckets; i++) materials[`materialTextures${i}`] = this.textureBuckets[i] ?? this.textureBuckets[0]!;
        (materials["materialTextureUvScale"] as ShaderVar)["tex"] = this.texInfoTexture;
        materials["webfalcorDummyTexture"] = this.dummyTexture;
        try {
            materials["webfalcorDisplacementTexture"] = this.displacementTexture ?? this.dummyTexture;
        } catch {
            /* displacement member absent in this kernel variant */
        }
        materials["materialBuffer0"] = this.buffers["materialBuffer0"]!;
        materials["materialTexture3D0"] = this.texture3D;
    }

    /**
     * HitInfo::init's bit allocation (128-bit format): hit type, instance ID over every geometry
     * instance, primitive index over the largest mesh/curve (SDF grids keep their primitive-ID bits).
     */
    private hitInfoDefines(): { HIT_INFO_TYPE_BITS: number; HIT_INFO_INSTANCE_ID_BITS: number; HIT_INFO_PRIMITIVE_INDEX_BITS: number } {
        const allocateBits = (count: number) => (count <= 1 ? 0 : Math.floor(Math.log2(count - 1)) + 1);
        const kHitTypeCount = 7;
        const instances = this.instanceCount + this.sdfGrids.length + this.sceneCurves.length + this.customPrimitives.length;
        let primitiveBits = allocateBits(this.maxPrimitiveCount);
        for (const d of this.sdfGrids) primitiveBits = Math.max(primitiveBits, 12, "maxPrimitiveIDBits" in d.grid ? (d.grid as { maxPrimitiveIDBits: number }).maxPrimitiveIDBits : 0);
        // Packed SBS grids hit with global brick IDs.
        const sbsBricks = [...new Set(this.sdfGrids.map((d) => d.grid))].reduce((n, g) => n + (g instanceof SDFSBS ? g.brickCount : 0), 0);
        if (sbsBricks > 0) primitiveBits = Math.max(primitiveBits, allocateBits(sbsBricks));
        const typeBits = allocateBits(kHitTypeCount);
        const instanceBits = allocateBits(instances);
        if (primitiveBits > 32 || typeBits + instanceBits > 32) throw new RuntimeError("Scene requires > 64 bits for encoding hit info header. This is currently not supported.");
        return { HIT_INFO_TYPE_BITS: typeBits, HIT_INFO_INSTANCE_ID_BITS: instanceBits, HIT_INFO_PRIMITIVE_INDEX_BITS: primitiveBits };
    }

    /** Mirrors Scene::setCameraControlsEnabled. */
    setCameraControlsEnabled(enabled: boolean): void {
        this.cameraControlsEnabled = enabled;
    }

    /** Mirrors getGeometryInstanceIDsByType(SDFGrid): one instance per SDF grid, after the meshes. */
    getSDFGridInstanceIDs(): number[] {
        return this.sdfGrids.map((_g, i) => this.sdfInstanceBase + i);
    }

    /** Mirrors findSDFGridIDFromGeometryInstanceID; -1 if the instance is not an SDF grid. */
    findSDFGridIDFromGeometryInstanceID(instanceID: number): number {
        const id = instanceID - this.sdfInstanceBase;
        return id >= 0 && id < this.sdfGrids.length ? id : -1;
    }

    /** The SDF grid instance's world matrix (AnimationController::getGlobalMatrices()[globalMatrixID]). */
    getSDFGridTransform(gridID: number): float4x4 {
        return this.sdfGrids[gridID]!.transform ?? float4x4.identity();
    }

    /**
     * Mirrors updateNodeTransform for an SDF grid instance: rewrites its world and
     * inverse-transpose matrices and drops the SDF acceleration data built from them.
     */
    updateSDFGridTransform(gridID: number, transform: float4x4): void {
        this.sdfGrids[gridID]!.transform = transform;
        const node = this.sdfInstanceBase + gridID;
        const buffer = this.buffers["worldMatrices"];
        if (buffer) {
            buffer.setBlob(new Float32Array(transform.toArray()), node * 64);
            buffer.setBlob(new Float32Array(transpose(inverse(transform)).toArray()), (this.invTransposeOffset + node) * 64);
        }
        this.invalidateSDFResources();
    }

    getGeometryInstanceCount(): number {
        return this.instanceCount;
    }

    /**
     * Native mesh ID (GeometryInstanceData::geometryID) for each triangle instance.
     * Instances of one mesh share it, unlike the web's one-mesh-per-instance layout.
     */
    getMeshIDs(): Uint32Array {
        return this.meshIDs;
    }

    /** Raster draw data (mirrors Scene::rasterize): buffers + per-mesh indexed draws. */
    getMeshDrawData(): {
        vertexBuffer: Buffer;
        drawIDBuffer: Buffer;
        indexBuffer: Buffer;
        draws: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[];
    } {
        return {
            vertexBuffer: this.buffers["vertices"]!,
            drawIDBuffer: this.buffers["drawIDs"]!,
            indexBuffer: this.buffers["indices"]!,
            draws: this.drawList,
        };
    }
}
