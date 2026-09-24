/**
 * Binary scene cache mirroring Scene/SceneCache.h in role: caches the
 * imported scene description so reloads skip script execution, importers,
 * and mesh processing. Web divergences (docs §9): OPFS storage keyed by a
 * SHA-256 of the scene source (no file timestamps). v4 covers every scene
 * class: geometry (incl. skin/morph data), materials, textures (original
 * bytes or lossless PNG), lights, camera, curves, env map (original bytes),
 * node animations + morph weight tracks, SDF grids (as rebuildable recipes)
 * and grid volumes (NanoVDB buffers).
 */

import { Camera } from "./Camera/Camera.js";
import type { Device } from "../Core/API/Device.js";
import { RuntimeError } from "../Core/Error.js";
import { float2, float3, float4 } from "../Utils/Math/Vector.js";
import { float4x4 } from "../Utils/Math/Matrix.js";
import { quatf } from "../Utils/Math/Quaternion.js";
import { Scene, type SceneMeshDesc, type SceneMaterialDesc, type SceneCurveDesc, type SceneMetadata } from "./Scene.js";
import { TextureManager, type TextureSource } from "./Material/TextureManager.js";
import { EnvMap } from "./Lights/EnvMap.js";
import type { AnalyticLight, StaticVertex } from "./SceneData.js";
import type { AnimationChannel, MorphDesc, SceneNode, SkinDesc, WeightTrack } from "./Animation/SceneAnimation.js";
import { buildSDFGridFromRecipe, type SDFGridRecipe } from "./SDFs/SDFGridRecipe.js";

/** A recipe in the cache header: the corner values move to the word blobs. */
type RecipeMeta = Omit<SDFGridRecipe, "ops"> & {
    ops: (Exclude<SDFGridRecipe["ops"][number], { kind: "values" }> | { kind: "values"; gridWidth: number; valueCount: number })[];
};
import type { SceneSDFGridDesc } from "./Scene.js";
import { GridVolume, type GridSlot } from "./Volume/GridVolume.js";
import { Grid } from "./Volume/Grid.js";

const kMagic = 0x43534657; // 'WFSC'
const kVersion = 6; // v4: + animation, skin/morph, SDF recipes, grid volumes; v5: camera list; v6: DDS textures, metadata
const kFloatsPerVertex = 13; // pos3 + normal3 + tangent4 + texCrd2 + curveRadius

export interface SceneCameraPose {
    position: [number, number, number];
    target: [number, number, number];
    up: [number, number, number];
    focalLength: number;
    focalDistance: number;
    apertureRadius: number;
    shutterSpeed?: number;
    ISOSpeed?: number;
    name?: string;
    depthRange?: [number, number];
    aspectRatio?: number;
}

export interface CacheableScene {
    meshes: SceneMeshDesc[];
    materials: SceneMaterialDesc[];
    lights: AnalyticLight[];
    nodes: SceneNode[];
    cameraNodeID?: number;
    /** Every scene camera; `selectedCamera` indexes it, `animatedCamera` is the one cameraNodeID drives. */
    cameras: SceneCameraPose[];
    selectedCamera: number;
    animatedCamera: number;
    /** Scene::Metadata and the camera speed. */
    metadata?: SceneMetadata;
    cameraSpeed?: number;
    /** Material textures as lossless PNG (phase 2). */
    textures: CachedTexture[];
    /** Static curve geometry (phase 3). */
    curves: SceneCurveDesc[];
    /** Env map as the original encoded .hdr/.exr file (phase 3). */
    envMap?: { bytes: Uint8Array; isExr: boolean; intensity: number; tint: [number, number, number]; rotationDeg: [number, number, number]; equalAreaOctahedral?: boolean };
    /** Node animation channels + morph weight tracks (v4). */
    animations: AnimationChannel[];
    weightTracks: WeightTrack[];
    /** SDF grids as recipes (rebuilt deterministically) + instances (v4). */
    sdfGrids: { recipes: SDFGridRecipe[]; instances: { gridIndex: number; materialID: number; transform?: float4x4 }[] };
    /** Grid volumes with their NanoVDB buffers (v4). */
    gridVolumes: CachedGridVolume[];
    /** SceneBuilder::addCustomPrimitive entries (user ID + AABB). */
    customPrimitives?: { userID: number; aabb: { min: [number, number, number]; max: [number, number, number] } }[];
}

export interface CachedGridVolume {
    name: string;
    densityScale: number;
    emissionScale: number;
    albedo: [number, number, number];
    anisotropy: number;
    emissionTemperature: number;
    grids: { slot: GridSlot; bytes: Uint8Array }[];
}

/** Plain-data snapshot of a scene's grid volumes for the cache. */
export function snapshotGridVolumes(scene: Scene): CachedGridVolume[] {
    return scene.gridVolumes.map((v) => ({
        name: v.name,
        densityScale: v.densityScale,
        emissionScale: v.emissionScale,
        albedo: [v.albedo.x, v.albedo.y, v.albedo.z],
        anisotropy: v.anisotropy,
        emissionTemperature: v.emissionTemperature,
        grids: (["density", "emission"] as GridSlot[]).flatMap((slot) => {
            const g = v.getGrid(slot);
            return g ? [{ slot, bytes: g.gridBuffer }] : [];
        }),
    }));
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

interface MeshMeta {
    materialID: number;
    nodeID?: number;
    transform?: { __m4: number[] };
    vertexCount: number;
    indexCount: number;
    skin?: { boneNodeIDs: number[]; inverseBind: unknown; count: number };
    morph?: { nodeID: number; baseWeights: number[]; targets: { posCount: number; normalCount: number }[] };
}

interface TrackMeta {
    nodeID: number;
    timesCount: number;
    valuesCount: number;
    interp: string;
    path?: string;
    clip?: number;
    preInfinity?: number;
    postInfinity?: number;
    numTargets?: number;
}

/** Word-aligned typed-array payload of the mesh/curve/animation classes, in file order. */
function wordBlobs(cached: CacheableScene): (Float32Array | Uint32Array)[] {
    const blobs: (Float32Array | Uint32Array)[] = [];
    for (const m of cached.meshes) {
        const verts = new Float32Array(m.vertices.length * kFloatsPerVertex);
        m.vertices.forEach((v, i) => {
            verts.set([v.position.x, v.position.y, v.position.z, v.normal.x, v.normal.y, v.normal.z, v.tangent.x, v.tangent.y, v.tangent.z, v.tangent.w, v.texCrd.x, v.texCrd.y, v.curveRadius ?? 0], i * kFloatsPerVertex);
        });
        blobs.push(verts, m.indices);
        if (m.skin) blobs.push(m.skin.boneIDs, m.skin.weights);
        if (m.morph) for (const t of m.morph.targets) blobs.push(t.position, t.normal ?? new Float32Array(0));
    }
    for (const c of cached.curves) blobs.push(c.positionsRadii, c.texCrds ?? new Float32Array(0), c.indices);
    for (const a of cached.animations) blobs.push(a.times, a.values);
    for (const w of cached.weightTracks) blobs.push(w.times, w.values);
    // SDF corner values (grids loaded from `.sdfg`): bulk float data, so they
    // ride in the blobs rather than the JSON header.
    for (const r of cached.sdfGrids.recipes) for (const op of r.ops) if (op.kind === "values") blobs.push(op.values);
    return blobs;
}

/** Byte-granular payload (compressed images, env map, NanoVDB buffers), in file order. */
function byteBlobs(cached: CacheableScene): Uint8Array[] {
    const blobs: Uint8Array[] = cached.textures.map((t) => t.png);
    if (cached.envMap) blobs.push(cached.envMap.bytes);
    for (const v of cached.gridVolumes) for (const g of v.grids) blobs.push(g.bytes);
    return blobs;
}

export function serializeScene(cached: CacheableScene): Uint8Array {
    const meshMeta: MeshMeta[] = cached.meshes.map((m) => ({
        materialID: m.materialID,
        nodeID: m.nodeID,
        transform: m.transform ? { __m4: Array.from(m.transform.data) } : undefined,
        vertexCount: m.vertices.length,
        indexCount: m.indices.length,
        skin: m.skin ? { boneNodeIDs: m.skin.boneNodeIDs, inverseBind: encodeValue(m.skin.inverseBind), count: m.skin.boneIDs.length } : undefined,
        morph: m.morph
            ? { nodeID: m.morph.nodeID, baseWeights: m.morph.baseWeights, targets: m.morph.targets.map((t) => ({ posCount: t.position.length, normalCount: t.normal?.length ?? 0 })) }
            : undefined,
    }));
    const header = {
        meshes: meshMeta,
        materials: encodeValue(cached.materials),
        lights: encodeValue(cached.lights),
        nodes: encodeValue(cached.nodes),
        cameraNodeID: cached.cameraNodeID,
        cameras: cached.cameras,
        selectedCamera: cached.selectedCamera,
        animatedCamera: cached.animatedCamera,
        metadata: cached.metadata,
        cameraSpeed: cached.cameraSpeed,
        textures: cached.textures.map((t) => ({ srgb: t.srgb, byteLength: t.png.byteLength, dds: t.dds })),
        curves: cached.curves.map((c) => ({
            floatCount: c.positionsRadii.length,
            texCrdCount: c.texCrds?.length ?? 0,
            indexCount: c.indices.length,
            materialID: c.materialID,
            transform: c.transform ? { __m4: Array.from(c.transform.data) } : undefined,
        })),
        envMap: cached.envMap
            ? {
                  byteLength: cached.envMap.bytes.byteLength,
                  isExr: cached.envMap.isExr,
                  intensity: cached.envMap.intensity,
                  tint: cached.envMap.tint,
                  rotationDeg: cached.envMap.rotationDeg,
                  equalAreaOctahedral: cached.envMap.equalAreaOctahedral,
              }
            : undefined,
        animations: cached.animations.map(
            (a): TrackMeta => ({ nodeID: a.nodeID, path: a.path, interp: a.interp, clip: a.clip, preInfinity: a.preInfinity, postInfinity: a.postInfinity, timesCount: a.times.length, valuesCount: a.values.length }),
        ),
        weightTracks: cached.weightTracks.map((w): TrackMeta => ({ nodeID: w.nodeID, numTargets: w.numTargets, interp: w.interp, timesCount: w.times.length, valuesCount: w.values.length })),
        sdfGrids: {
            recipes: cached.sdfGrids.recipes.map((r) => ({
                ...r,
                ops: r.ops.map((op) => (op.kind === "values" ? { kind: op.kind, gridWidth: op.gridWidth, valueCount: op.values.length } : op)),
            })),
            instances: cached.sdfGrids.instances.map((i) => ({ gridIndex: i.gridIndex, materialID: i.materialID, transform: i.transform ? { __m4: Array.from(i.transform.data) } : undefined })),
        },
        gridVolumes: cached.gridVolumes.map((v) => ({ ...v, grids: v.grids.map((g) => ({ slot: g.slot, byteLength: g.bytes.byteLength })) })),
        customPrimitives: cached.customPrimitives ?? [],
    };
    const json = new TextEncoder().encode(JSON.stringify(header));
    const jsonPadded = (json.length + 3) & ~3;

    const words = wordBlobs(cached);
    const bytesList = byteBlobs(cached);
    const wordBytes = words.reduce((acc, b) => acc + b.length * 4, 0);
    const byteBytes = bytesList.reduce((acc, b) => acc + b.byteLength, 0);
    const out = new Uint8Array(12 + jsonPadded + wordBytes + byteBytes);
    const dv = new DataView(out.buffer);
    dv.setUint32(0, kMagic, true);
    dv.setUint32(4, kVersion, true);
    dv.setUint32(8, json.length, true);
    out.set(json, 12);

    let off = 12 + jsonPadded;
    for (const b of words) {
        // Typed views need 4-byte alignment; `off` is word-aligned here by construction.
        out.set(new Uint8Array(b.buffer, b.byteOffset, b.length * 4), off);
        off += b.length * 4;
    }
    for (const b of bytesList) {
        out.set(b, off);
        off += b.byteLength;
    }
    return out;
}

export function deserializeScene(bytes: Uint8Array): CacheableScene {
    const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    if (dv.getUint32(0, true) !== kMagic || dv.getUint32(4, true) !== kVersion) throw new RuntimeError("SceneCache: bad magic/version");
    const jsonLen = dv.getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + jsonLen))) as {
        meshes: MeshMeta[];
        materials: unknown;
        lights: unknown;
        nodes: unknown;
        cameraNodeID?: number;
        cameras: SceneCameraPose[];
        selectedCamera: number;
        animatedCamera: number;
        metadata?: SceneMetadata;
        cameraSpeed?: number;
        textures: { srgb: boolean; byteLength: number; dds?: boolean }[];
        curves: { floatCount: number; texCrdCount: number; indexCount: number; materialID: number; transform?: { __m4: number[] } }[];
        envMap?: { byteLength: number; isExr: boolean; intensity: number; tint: [number, number, number]; rotationDeg: [number, number, number]; equalAreaOctahedral?: boolean };
        animations: TrackMeta[];
        weightTracks: TrackMeta[];
        sdfGrids: { recipes: RecipeMeta[]; instances: { gridIndex: number; materialID: number; transform?: { __m4: number[] } }[] };
        gridVolumes: (Omit<CachedGridVolume, "grids"> & { grids: { slot: GridSlot; byteLength: number }[] })[];
        customPrimitives?: CacheableScene["customPrimitives"];
    };

    let off = 12 + ((jsonLen + 3) & ~3);
    // Views bounded to whole words (the trailing byte blobs have arbitrary length).
    const words = Math.floor(bytes.byteLength / 4);
    const f32 = new Float32Array(bytes.buffer, bytes.byteOffset, words);
    const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, words);
    const takeF32 = (count: number): Float32Array => {
        const a = new Float32Array(f32.subarray(off / 4, off / 4 + count));
        off += count * 4;
        return a;
    };
    const takeU32 = (count: number): Uint32Array => {
        const a = new Uint32Array(u32.subarray(off / 4, off / 4 + count));
        off += count * 4;
        return a;
    };
    const takeBytes = (count: number): Uint8Array => {
        const a = bytes.slice(off, off + count);
        off += count;
        return a;
    };
    const mat4 = (m?: { __m4: number[] }) => (m ? new float4x4(new Float32Array(m.__m4)) : undefined);

    const meshes: SceneMeshDesc[] = header.meshes.map((meta) => {
        const verts = takeF32(meta.vertexCount * kFloatsPerVertex);
        const vertices: StaticVertex[] = [];
        for (let v = 0; v < meta.vertexCount; v++) {
            const fi = v * kFloatsPerVertex;
            vertices.push({
                position: new float3(verts[fi]!, verts[fi + 1]!, verts[fi + 2]!),
                normal: new float3(verts[fi + 3]!, verts[fi + 4]!, verts[fi + 5]!),
                tangent: new float4(verts[fi + 6]!, verts[fi + 7]!, verts[fi + 8]!, verts[fi + 9]!),
                texCrd: new float2(verts[fi + 10]!, verts[fi + 11]!),
                curveRadius: verts[fi + 12]!,
            });
        }
        const indices = takeU32(meta.indexCount);
        let skin: SkinDesc | undefined;
        if (meta.skin) {
            skin = { boneNodeIDs: meta.skin.boneNodeIDs, inverseBind: decodeValue(meta.skin.inverseBind) as float4x4[], boneIDs: takeU32(meta.skin.count), weights: takeF32(meta.skin.count) };
        }
        let morph: MorphDesc | undefined;
        if (meta.morph) {
            morph = {
                nodeID: meta.morph.nodeID,
                baseWeights: meta.morph.baseWeights,
                targets: meta.morph.targets.map((t) => {
                    const position = takeF32(t.posCount);
                    const normal = t.normalCount > 0 ? takeF32(t.normalCount) : undefined;
                    return normal ? { position, normal } : { position };
                }),
            };
        }
        return { vertices, indices, materialID: meta.materialID, nodeID: meta.nodeID, transform: mat4(meta.transform), skin, morph };
    });

    const curves: SceneCurveDesc[] = header.curves.map((meta) => {
        const positionsRadii = takeF32(meta.floatCount);
        const texCrds = meta.texCrdCount > 0 ? takeF32(meta.texCrdCount) : null;
        const indices = takeU32(meta.indexCount);
        return { positionsRadii, texCrds, indices, materialID: meta.materialID, transform: mat4(meta.transform) };
    });

    const animations: AnimationChannel[] = header.animations.map((meta) => ({
        nodeID: meta.nodeID,
        path: meta.path as AnimationChannel["path"],
        times: takeF32(meta.timesCount),
        values: takeF32(meta.valuesCount),
        interp: meta.interp as AnimationChannel["interp"],
        clip: meta.clip,
        preInfinity: meta.preInfinity,
        postInfinity: meta.postInfinity,
    }));
    const weightTracks: WeightTrack[] = header.weightTracks.map((meta) => ({
        nodeID: meta.nodeID,
        times: takeF32(meta.timesCount),
        values: takeF32(meta.valuesCount),
        numTargets: meta.numTargets ?? 0,
        interp: meta.interp as WeightTrack["interp"],
    }));

    // Word blobs are consumed in wordBlobs() order: the SDF values come last.
    const sdfRecipes: SDFGridRecipe[] = header.sdfGrids.recipes.map((r) => ({
        ...r,
        ops: r.ops.map((op) => (op.kind === "values" ? { kind: op.kind, gridWidth: op.gridWidth, values: takeF32(op.valueCount) } : op)),
    }));

    const textures = header.textures.map((meta) => ({ png: takeBytes(meta.byteLength), srgb: meta.srgb, dds: meta.dds }));
    let envMap: CacheableScene["envMap"];
    if (header.envMap) {
        envMap = { bytes: takeBytes(header.envMap.byteLength), isExr: header.envMap.isExr, intensity: header.envMap.intensity, tint: header.envMap.tint, rotationDeg: header.envMap.rotationDeg, equalAreaOctahedral: header.envMap.equalAreaOctahedral };
    }
    const gridVolumes: CachedGridVolume[] = header.gridVolumes.map((v) => ({ ...v, grids: v.grids.map((g) => ({ slot: g.slot, bytes: takeBytes(g.byteLength) })) }));

    return {
        meshes,
        materials: decodeValue(header.materials) as SceneMaterialDesc[],
        lights: decodeValue(header.lights) as AnalyticLight[],
        nodes: decodeValue(header.nodes) as SceneNode[],
        cameraNodeID: header.cameraNodeID,
        cameras: header.cameras,
        selectedCamera: header.selectedCamera,
        animatedCamera: header.animatedCamera,
        metadata: header.metadata,
        cameraSpeed: header.cameraSpeed,
        textures,
        curves,
        envMap,
        animations,
        weightTracks,
        sdfGrids: { recipes: sdfRecipes, instances: header.sdfGrids.instances.map((i) => ({ gridIndex: i.gridIndex, materialID: i.materialID, transform: mat4(i.transform) })) },
        gridVolumes,
        customPrimitives: header.customPrimitives ?? [],
    };
}

/** A cached material texture: browser-decodable image bytes (PNG etc.) or, with `dds`, a DDS file. */
export interface CachedTexture {
    png: Uint8Array;
    srgb: boolean;
    dds?: boolean;
}

/** PNG, JPEG, WebP, GIF or BMP magic (what createImageBitmap decodes). */
function browserDecodable(bytes: Uint8Array): boolean {
    const b = (i: number) => bytes[i] ?? 0;
    return (
        (b(0) === 0x89 && b(1) === 0x50) || (b(0) === 0xff && b(1) === 0xd8) || (b(0) === 0x52 && b(8) === 0x57) || (b(0) === 0x47 && b(1) === 0x49) || (b(0) === 0x42 && b(1) === 0x4d)
    );
}

/** Collects texture sources for the cache: the original bytes when retained and
 *  decodable here again (images, DDS), else a PNG re-encode (canvas roundtrip;
 *  premultiply can differ by 1 lsb on translucent pixels). */
export async function encodeTextureSources(textureManager: TextureManager): Promise<CachedTexture[]> {
    const out: CachedTexture[] = [];
    for (let i = 0; i < textureManager.count; i++) {
        const source = textureManager.getSource(i)!;
        if (source.bytes && (source.dds || browserDecodable(source.bytes))) {
            out.push({ png: source.bytes, srgb: source.srgb, dds: source.dds || undefined });
            continue;
        }
        const canvas = new OffscreenCanvas(source.bitmap.width, source.bitmap.height);
        canvas.getContext("2d")!.drawImage(source.bitmap, 0, 0);
        const blob = await canvas.convertToBlob({ type: "image/png" });
        out.push({ png: new Uint8Array(await blob.arrayBuffer()), srgb: source.srgb });
    }
    return out;
}

async function decodeTextureSources(textures: CachedTexture[]): Promise<TextureManager> {
    const tm = new TextureManager();
    for (const t of textures) {
        if (t.dds) {
            // DDS: the BC chain for the GPU and the capped decode for CPU analysis, as on import.
            const { ddsCompressedPayload, decodeDDSToRGBA } = await import("./Importer/DDSLoader.js");
            const buffer = t.png.slice().buffer as ArrayBuffer;
            const { width, height, rgba } = decodeDDSToRGBA(buffer, t.srgb, 512);
            const bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(rgba), width, height));
            tm.addTexture({ bitmap, srgb: t.srgb, bytes: t.png, compressed: ddsCompressedPayload(buffer, t.srgb), dds: true });
            continue;
        }
        // Same decode options as the pyscene import path (parity-critical).
        const bitmap = await createImageBitmap(new Blob([t.png.slice().buffer as ArrayBuffer]), { colorSpaceConversion: "none" });
        tm.addTexture({ bitmap, srgb: t.srgb, bytes: t.png } as TextureSource);
    }
    return tm;
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
export async function buildSceneFromCache(device: Device, cached: CacheableScene): Promise<Scene> {
    const textureManager = await decodeTextureSources(cached.textures);
    // SDF grids are rebuilt from their recipes (deterministic generators), shared across instances.
    const builtGrids = cached.sdfGrids.recipes.map(buildSDFGridFromRecipe);
    const sdfGrids: SceneSDFGridDesc[] = cached.sdfGrids.instances.map((i) => ({ grid: builtGrids[i.gridIndex]!, materialID: i.materialID, transform: i.transform }));
    const scene = new Scene(device, cached.meshes, cached.materials, cached.lights, textureManager, sdfGrids, cached.nodes, cached.animations, cached.cameraNodeID, cached.weightTracks, cached.curves);
    for (const c of cached.customPrimitives ?? []) scene.addCustomPrimitive(c.userID, c.aabb);
    for (const v of cached.gridVolumes) {
        const vol = new GridVolume(v.name);
        vol.densityScale = v.densityScale;
        vol.emissionScale = v.emissionScale;
        vol.albedo = new float3(...v.albedo);
        vol.anisotropy = v.anisotropy;
        vol.emissionTemperature = v.emissionTemperature;
        for (const g of v.grids) vol.setGrid(g.slot, new Grid(device, g.bytes));
        scene.gridVolumes.push(vol);
    }
    scene.finalizeGridVolumes();
    if (cached.envMap) {
        const env = EnvMap.createFromBytes(device, cached.envMap.bytes, cached.envMap.isExr, { equalAreaOctahedral: cached.envMap.equalAreaOctahedral });
        env.intensity = cached.envMap.intensity;
        env.tint = cached.envMap.tint;
        env.setRotation(cached.envMap.rotationDeg);
        scene.setEnvMap(env);
    }
    const cameras = cached.cameras.map((pose) => {
        const cam = new Camera(pose.name ?? "Camera");
        cam.setPosition(new float3(...pose.position));
        cam.setTarget(new float3(...pose.target));
        cam.setUpVector(new float3(...pose.up));
        cam.setFocalLength(pose.focalLength);
        cam.setFocalDistance(pose.focalDistance);
        cam.setApertureRadius(pose.apertureRadius);
        if (pose.shutterSpeed !== undefined) cam.setShutterSpeed(pose.shutterSpeed);
        if (pose.ISOSpeed !== undefined) cam.setISOSpeed(pose.ISOSpeed);
        if (pose.depthRange) cam.setDepthRange(...pose.depthRange);
        if (pose.aspectRatio) cam.setAspectRatio(pose.aspectRatio);
        return cam;
    });
    scene.setCameraList(cameras, cached.selectedCamera, cached.animatedCamera);
    scene.metadata = { ...(cached.metadata ?? {}) };
    if (cached.cameraSpeed !== undefined) scene.cameraSpeed = cached.cameraSpeed;
    return scene;
}

/** Camera snapshots for the cache (read back off the built scene). */
export function snapshotCameras(scene: Scene): Pick<CacheableScene, "cameras" | "selectedCamera" | "animatedCamera" | "metadata" | "cameraSpeed"> {
    const cameras = scene.getCameras().map((c): SceneCameraPose => {
        const p = c.getPosition();
        const t = c.getTarget();
        const u = c.getUpVector();
        return {
            name: c.name,
            position: [p.x, p.y, p.z],
            target: [t.x, t.y, t.z],
            up: [u.x, u.y, u.z],
            focalLength: c.getFocalLength(),
            focalDistance: c.getFocalDistance(),
            apertureRadius: c.getApertureRadius(),
            shutterSpeed: c.getShutterSpeed(),
            ISOSpeed: c.getISOSpeed(),
            depthRange: [c.getNearPlane(), c.getFarPlane()],
            aspectRatio: c.getAspectRatio(),
        };
    });
    return { cameras, selectedCamera: scene.getSelectedCameraIndex(), animatedCamera: scene.getAnimatedCameraIndex(), metadata: { ...scene.metadata }, cameraSpeed: scene.cameraSpeed };
}
