/**
 * Binary scene cache mirroring Scene/SceneCache.h in role: caches the
 * imported scene description so reloads skip script execution, importers,
 * and mesh processing. Web divergences (docs §9): OPFS storage keyed by a
 * SHA-256 of the scene source (no file timestamps); phase 1 covers static
 * geometry + materials + analytic lights + camera — scenes with textures,
 * env maps, volumes, SDF grids, curves, or animation fall back to import.
 */

import type { Device } from "../Core/API/Device.js";
import { RuntimeError } from "../Core/Error.js";
import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { float4x4 } from "../Utils/Math/Matrix.js";
import { quatf } from "../Utils/Math/Quaternion.js";
import { Scene, type SceneMeshDesc, type SceneMaterialDesc } from "./Scene.js";
import { TextureManager } from "./Material/TextureManager.js";
import type { AnalyticLight, StaticVertex } from "./SceneData.js";
import type { SceneNode } from "./Animation/SceneAnimation.js";

const kMagic = 0x43534657; // 'WFSC'
const kVersion = 1;
const kFloatsPerVertex = 13; // pos3 + normal3 + tangent4 + texCrd2 + curveRadius

export interface SceneCameraPose {
    position: [number, number, number];
    target: [number, number, number];
    up: [number, number, number];
    focalLength: number;
    focalDistance: number;
    apertureRadius: number;
}

export interface CacheableScene {
    meshes: SceneMeshDesc[];
    materials: SceneMaterialDesc[];
    lights: AnalyticLight[];
    nodes: SceneNode[];
    cameraNodeID?: number;
    camera: SceneCameraPose;
}

/** Tags math types so plain JSON survives the round trip. */
function encodeValue(v: unknown): unknown {
    if (v instanceof float2) return { __f2: [v.x, v.y] };
    if (v instanceof float3) return { __f3: [v.x, v.y, v.z] };
    if (v instanceof float4) return { __f4: [v.x, v.y, v.z, v.w] };
    if (v instanceof quatf) return { __q: [v.x, v.y, v.z, v.w] };
    if (v instanceof float4x4) return { __m4: Array.from(v.data) };
    if (Array.isArray(v)) return v.map(encodeValue);
    if (v !== null && typeof v === "object") {
        return Object.fromEntries(Object.entries(v).map(([k, val]) => [k, encodeValue(val)]));
    }
    return v;
}

function decodeValue(v: unknown): unknown {
    if (Array.isArray(v)) return v.map(decodeValue);
    if (v !== null && typeof v === "object") {
        const o = v as Record<string, number[]>;
        if (o["__f2"]) return new float2(o["__f2"][0]!, o["__f2"][1]!);
        if (o["__f3"]) return new float3(o["__f3"][0]!, o["__f3"][1]!, o["__f3"][2]!);
        if (o["__f4"]) return new float4(o["__f4"][0]!, o["__f4"][1]!, o["__f4"][2]!, o["__f4"][3]!);
        if (o["__q"]) return new quatf(o["__q"][0]!, o["__q"][1]!, o["__q"][2]!, o["__q"][3]!);
        if (o["__m4"]) return new float4x4(new Float32Array(o["__m4"]));
        return Object.fromEntries(Object.entries(v).map(([k, val]) => [k, decodeValue(val)]));
    }
    return v;
}

/** Cache key: SHA-256 of the scene source text. */
export async function sceneCacheKey(source: string): Promise<string> {
    const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export function serializeScene(cached: CacheableScene): Uint8Array {
    const meshMeta = cached.meshes.map((m) => ({
        materialID: m.materialID,
        nodeID: m.nodeID,
        transform: m.transform ? { __m4: Array.from(m.transform.data) } : undefined,
        vertexCount: m.vertices.length,
        indexCount: m.indices.length,
    }));
    const header = {
        meshes: meshMeta,
        materials: encodeValue(cached.materials),
        lights: encodeValue(cached.lights),
        nodes: encodeValue(cached.nodes),
        cameraNodeID: cached.cameraNodeID,
        camera: cached.camera,
    };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const jsonPadded = (json.length + 3) & ~3;

    let blobFloats = 0;
    for (const m of cached.meshes) blobFloats += m.vertices.length * kFloatsPerVertex + m.indices.length;
    const total = 12 + jsonPadded + blobFloats * 4;
    const out = new Uint8Array(total);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, kMagic, true);
    dv.setUint32(4, kVersion, true);
    dv.setUint32(8, json.length, true);
    out.set(json, 12);

    let off = 12 + jsonPadded;
    const f32 = new Float32Array(out.buffer);
    const u32 = new Uint32Array(out.buffer);
    for (const m of cached.meshes) {
        for (const v of m.vertices) {
            let fi = off / 4;
            f32[fi++] = v.position.x; f32[fi++] = v.position.y; f32[fi++] = v.position.z;
            f32[fi++] = v.normal.x; f32[fi++] = v.normal.y; f32[fi++] = v.normal.z;
            f32[fi++] = v.tangent.x; f32[fi++] = v.tangent.y; f32[fi++] = v.tangent.z; f32[fi++] = v.tangent.w;
            f32[fi++] = v.texCrd.x; f32[fi++] = v.texCrd.y;
            f32[fi++] = v.curveRadius ?? 0;
            off += kFloatsPerVertex * 4;
        }
        u32.set(m.indices, off / 4);
        off += m.indices.length * 4;
    }
    return out;
}

export function deserializeScene(bytes: Uint8Array): CacheableScene {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.getUint32(0, true) !== kMagic || dv.getUint32(4, true) !== kVersion) throw new RuntimeError("SceneCache: bad magic/version");
    const jsonLen = dv.getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + jsonLen))) as {
        meshes: { materialID: number; nodeID?: number; transform?: { __m4: number[] }; vertexCount: number; indexCount: number }[];
        materials: unknown;
        lights: unknown;
        nodes: unknown;
        cameraNodeID?: number;
        camera: SceneCameraPose;
    };

    let off = 12 + ((jsonLen + 3) & ~3);
    const f32 = new Float32Array(bytes.buffer, bytes.byteOffset);
    const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset);
    const meshes: SceneMeshDesc[] = header.meshes.map((meta) => {
        const vertices: StaticVertex[] = [];
        for (let v = 0; v < meta.vertexCount; v++) {
            let fi = off / 4;
            vertices.push({
                position: new float3(f32[fi]!, f32[fi + 1]!, f32[fi + 2]!),
                normal: new float3(f32[fi + 3]!, f32[fi + 4]!, f32[fi + 5]!),
                tangent: new float4(f32[fi + 6]!, f32[fi + 7]!, f32[fi + 8]!, f32[fi + 9]!),
                texCrd: new float2(f32[fi + 10]!, f32[fi + 11]!),
                curveRadius: f32[fi + 12]!,
            });
            off += kFloatsPerVertex * 4;
        }
        const indices = new Uint32Array(u32.subarray(off / 4, off / 4 + meta.indexCount));
        off += meta.indexCount * 4;
        return {
            vertices,
            indices,
            materialID: meta.materialID,
            nodeID: meta.nodeID,
            transform: meta.transform ? new float4x4(new Float32Array(meta.transform.__m4)) : undefined,
        };
    });

    return {
        meshes,
        materials: decodeValue(header.materials) as SceneMaterialDesc[],
        lights: decodeValue(header.lights) as AnalyticLight[],
        nodes: decodeValue(header.nodes) as SceneNode[],
        cameraNodeID: header.cameraNodeID,
        camera: header.camera,
    };
}

const kCachePrefix = "webfalcor-scene-";

export async function storeSceneCache(key: string, cached: CacheableScene): Promise<void> {
    const dir = await navigator.storage.getDirectory();
    const file = await dir.getFileHandle(`${kCachePrefix}${key}.bin`, { create: true });
    const writable = await file.createWritable();
    const bytes = serializeScene(cached);
    await writable.write(bytes.slice().buffer as ArrayBuffer);
    await writable.close();
}

export async function loadSceneCache(key: string): Promise<CacheableScene | null> {
    try {
        const dir = await navigator.storage.getDirectory();
        const file = await dir.getFileHandle(`${kCachePrefix}${key}.bin`);
        const bytes = new Uint8Array(await (await file.getFile()).arrayBuffer());
        return deserializeScene(bytes);
    } catch {
        return null;
    }
}

export async function clearSceneCache(): Promise<void> {
    const dir = await navigator.storage.getDirectory();
    const names: string[] = [];
    for await (const name of (dir as unknown as { keys(): AsyncIterable<string> }).keys()) {
        if (name.startsWith(kCachePrefix)) names.push(name);
    }
    for (const name of names) await dir.removeEntry(name);
}

/** Rebuilds a Scene from cached data (the fast-reload path). */
export function buildSceneFromCache(device: Device, cached: CacheableScene): Scene {
    const scene = new Scene(device, cached.meshes, cached.materials, cached.lights, new TextureManager(), [], cached.nodes, [], cached.cameraNodeID, [], []);
    const cam = cached.camera;
    scene.camera.setPosition(new float3(...cam.position));
    scene.camera.setTarget(new float3(...cam.target));
    scene.camera.setUpVector(new float3(...cam.up));
    scene.camera.setFocalLength(cam.focalLength);
    scene.camera.setFocalDistance(cam.focalDistance);
    scene.camera.setApertureRadius(cam.apertureRadius);
    return scene;
}

/** Camera pose snapshot for the cache (read back off the built scene). */
export function snapshotCameraPose(scene: Scene): SceneCameraPose {
    const p = scene.camera.getPosition();
    const t = scene.camera.getTarget();
    const u = scene.camera.getUpVector();
    return {
        position: [p.x, p.y, p.z],
        target: [t.x, t.y, t.z],
        up: [u.x, u.y, u.z],
        focalLength: scene.camera.getFocalLength(),
        focalDistance: scene.camera.getFocalDistance(),
        apertureRadius: scene.camera.getApertureRadius(),
    };
}
