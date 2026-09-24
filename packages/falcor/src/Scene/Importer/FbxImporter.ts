/**
 * FBX importer mirroring plugins/importers/AssimpImporter (Default import
 * mode). Parsing runs through Assimp compiled to WASM (see below), emitting the
 * aiScene as JSON; this module ports the
 * native mapping: node-hierarchy flattening, Default-mode material semantics
 * (diffuse/specular/emissive colors, shininess into specular.a, opacity ->
 * specular transmission, ".DoubleSided" name suffix) and the Default-mode
 * texture-slot table (DIFFUSE->BaseColor, SPECULAR->Specular,
 * EMISSIVE->Emissive, NORMALS->Normal).
 *
 * Parsing runs native's Assimp version (5.2.5, compiled to wasm) with native's post-process
 * flags (aiProcessPreset_TargetRealtime_MaxQuality | FlipUVs | RemoveComponent, minus its
 * exclusions; DontMergeMeshes and UseOriginalTangentSpace honoured) and component removal.
 */

import { float2, float3, float4, normalize3, cross, sub3, add3 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { MaterialType, ShadingModel, packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import { getTextureSlotSrgb } from "../Material/TextureSlots.js";
import { WorkerPool } from "../../Utils/Threading/WorkerPool.js";
import { ddsCompressedPayload } from "./DDSLoader.js";
import type { SceneMaterialDesc, SceneMeshDesc } from "../Scene.js";
import { decomposeTRS, type SceneNode, type AnimationChannel, type SkinDesc } from "../Animation/SceneAnimation.js";
import { LightType, type AnalyticLight, type StaticVertex } from "../SceneData.js";
import type { TextureManager } from "../Material/TextureManager.js";
import { Camera } from "../Camera/Camera.js";
import type { GltfCameraPose } from "./GltfImporter.js";

/** A camera from an imported file; `nodeID` is set when an animated node drives it. */
export interface ImportedCamera {
    name: string;
    pose: GltfCameraPose;
    aspectRatio?: number;
    nodeID?: number;
}

interface AiProperty {
    key: string;
    semantic: number;
    index: number;
    type: number;
    value: unknown;
}

interface AiMaterial {
    properties: AiProperty[];
}

/** assjson bone: name matches a node; offsetmatrix is row-major (aiMatrix4x4);
 *  weights are [vertexId, weight] pairs into the (post-process) vertex array. */
interface AiBone {
    name: string;
    offsetmatrix: number[];
    weights: [number, number][];
}

interface AiMesh {
    name: string;
    materialindex: number;
    bones?: AiBone[];
    // From the binary mesh blob (import.cpp packMeshes), not the assjson text.
    vertices: Float32Array;
    normals?: Float32Array;
    texturecoords?: Float32Array[];
    numuvcomponents?: number[];
    /** Indices of the mesh's 3-index faces. */
    triangles: Uint32Array;
}

interface AiNode {
    name: string;
    transformation: number[];
    children?: AiNode[];
    meshes?: number[];
}

/** assjson keyframe: [timeInTicks, [values...]]. Rotation values are [w,x,y,z]. */
type AiKey = [number, number[]];
interface AiNodeAnim {
    name: string; // animated node's name
    positionkeys?: AiKey[];
    rotationkeys?: AiKey[];
    scalingkeys?: AiKey[];
}
interface AiAnimation {
    name: string;
    tickspersecond?: number;
    duration?: number;
    channels: AiNodeAnim[];
}

interface AiLight {
    name: string; // matches a node in the hierarchy (gives the light's world transform)
    type: number; // aiLightSource: 1=directional, 2=point, 3=spot
    diffusecolor?: number[];
    direction?: number[];
}

interface AiScene {
    rootnode: AiNode;
    meshes: AiMesh[];
    materials: AiMaterial[];
    animations?: AiAnimation[];
    lights?: AiLight[];
    cameras?: AiCamera[];
}

interface AiCamera {
    name: string;
    aspect?: number;
    clipplanenear?: number;
    clipplanefar?: number;
    position?: number[];
    up?: number[];
    lookat?: number[];
}

interface AssimpModule {
    HEAPU8: Uint8Array;
    _malloc(bytes: number): number;
    _free(ptr: number): void;
    _ai_clear_files(): void;
    _ai_add_file(name: number, data: number, size: number): void;
    _ai_import(mainFile: number, flags: number, removeComponents: number): number;
    _ai_result(): number;
    _ai_result_size(): number;
    _ai_mesh_blob(): number;
    _ai_mesh_blob_size(): number;
    _ai_error(): number;
    _ai_free_result(): void;
    UTF8ToString(ptr: number): string;
    stringToNewUTF8(s: string): number;
}

let assimpModule: Promise<AssimpModule> | null = null;

/** Assimp 5.2.5 (native's version) compiled to wasm (packages/falcor/wasm, scripts/build-assimp-wasm.mjs). */
function getAssimp(): Promise<AssimpModule> {
    // Literal URLs so bundlers emit both files; the wasm is located explicitly.
    const wasmUrl = new URL("../../../wasm/assimp.wasm", import.meta.url).href;
    assimpModule ??= import(/* @vite-ignore */ new URL("../../../wasm/assimp.mjs", import.meta.url).href).then((m: { default: (opts: object) => Promise<AssimpModule> }) =>
        m.default({ locateFile: () => wasmUrl }),
    );
    return assimpModule;
}

/** aiPostProcessSteps (assimp/postprocess.h). */
const aiProcess = {
    CalcTangentSpace: 0x1, JoinIdenticalVertices: 0x2, Triangulate: 0x8, RemoveComponent: 0x10, GenNormals: 0x20, GenSmoothNormals: 0x40,
    SplitLargeMeshes: 0x80, PreTransformVertices: 0x100, LimitBoneWeights: 0x200, ValidateDataStructure: 0x400, ImproveCacheLocality: 0x800,
    RemoveRedundantMaterials: 0x1000, SortByPType: 0x8000, FindDegenerates: 0x10000, FindInvalidData: 0x20000, GenUVCoords: 0x40000,
    FindInstances: 0x100000, OptimizeMeshes: 0x200000, OptimizeGraph: 0x400000, FlipUVs: 0x800000,
} as const;
const kTargetRealtimeMaxQuality =
    aiProcess.CalcTangentSpace | aiProcess.GenSmoothNormals | aiProcess.JoinIdenticalVertices | aiProcess.ImproveCacheLocality | aiProcess.LimitBoneWeights |
    aiProcess.RemoveRedundantMaterials | aiProcess.SplitLargeMeshes | aiProcess.Triangulate | aiProcess.GenUVCoords | aiProcess.SortByPType |
    aiProcess.FindDegenerates | aiProcess.FindInvalidData | aiProcess.FindInstances | aiProcess.ValidateDataStructure | aiProcess.OptimizeMeshes;

/** Runs assimp over `files` (the first is the scene) with the given flags and AI_CONFIG_PP_RVC_FLAGS; returns the assjson scene. */
async function assimpImport(files: { name: string; bytes: Uint8Array }[], flags: number, removeComponents: number, what: string): Promise<AiScene> {
    const ai = await getAssimp();
    ai._ai_clear_files();
    for (const f of files) {
        const name = ai.stringToNewUTF8(f.name);
        const data = ai._malloc(Math.max(1, f.bytes.length));
        ai.HEAPU8.set(f.bytes, data);
        ai._ai_add_file(name, data, f.bytes.length);
        ai._free(data);
        ai._free(name);
    }
    const main = ai.stringToNewUTF8(files[0]!.name);
    const ok = ai._ai_import(main, flags >>> 0, removeComponents >>> 0);
    ai._free(main);
    ai._ai_clear_files();
    if (!ok) throw new RuntimeError(`${what}: assimp failed (${ai.UTF8ToString(ai._ai_error())})`);
    const json = new TextDecoder().decode(ai.HEAPU8.subarray(ai._ai_result(), ai._ai_result() + ai._ai_result_size()));
    // Copy the mesh blob out of the wasm heap before freeing it.
    const blob = new Uint32Array(ai.HEAPU8.buffer.slice(ai._ai_mesh_blob(), ai._ai_mesh_blob() + ai._ai_mesh_blob_size() * 4));
    ai._ai_free_result();
    const scene = JSON.parse(json) as AiScene;
    attachMeshArrays(scene, blob);
    return scene;
}

/** Attaches the binary mesh arrays (layout: import.cpp packMeshes) as views into `blob`. */
function attachMeshArrays(scene: AiScene, blob: Uint32Array): void {
    const floats = new Float32Array(blob.buffer, blob.byteOffset, blob.length);
    let at = 0;
    const meshCount = blob[at++]!;
    for (let m = 0; m < meshCount; m++) {
        const n = blob[at++]!;
        const hasNormals = blob[at++]! !== 0;
        const channels = blob[at++]!;
        const comps = Array.from(blob.subarray(at, at + channels));
        at += channels;
        const triIndices = blob[at++]!;
        const mesh = scene.meshes[m]!;
        mesh.vertices = floats.subarray(at, (at += n * 3));
        mesh.normals = hasNormals ? floats.subarray(at, (at += n * 3)) : undefined;
        mesh.texturecoords = comps.map((c) => floats.subarray(at, (at += n * c)));
        mesh.numuvcomponents = comps;
        mesh.triangles = blob.subarray(at, (at += triIndices));
    }
}

function decodeProp(p: AiProperty): unknown {
    // assjson types: 1=float(s), 3=string, 4=int(s), 5=binary (base64 of raw bytes).
    if (p.type === 5 && typeof p.value === "string") {
        const bin = atob((p.value as string).trim());
        if (bin.length >= 4) {
            return (bin.charCodeAt(0) | (bin.charCodeAt(1) << 8) | (bin.charCodeAt(2) << 16) | (bin.charCodeAt(3) << 24)) >>> 0;
        }
        return 0;
    }
    return p.value;
}

function findProp(mat: AiMaterial, key: string, semantic = 0): unknown {
    const p = mat.properties.find((q) => q.key === key && (semantic === 0 || q.semantic === semantic));
    return p ? decodeProp(p) : undefined;
}

function textureFile(mat: AiMaterial, aiType: number): string | undefined {
    const p = mat.properties.find((q) => q.key === "$tex.file" && q.semantic === aiType && q.index === 0);
    return p ? String(p.value) : undefined;
}

/** Mirrors AssimpImporter's ImportMode: OBJ gets its own material semantics. */
enum ImportMode {
    Default,
    OBJ,
}

/** aiTextureType values the texture tables use. */
const aiTextureType = { DIFFUSE: 1, SPECULAR: 2, EMISSIVE: 4, HEIGHT: 5, NORMALS: 6, DISPLACEMENT: 9 } as const;

/**
 * Mirrors kTextureMappings (Default and OBJ; glTF has its own importer here).
 * Later entries win when two map onto the same slot, as they do natively.
 */
const kTextureMappings: Record<ImportMode, { aiType: number; slot: "BaseColor" | "Specular" | "Emissive" | "Normal" }[]> = {
    [ImportMode.Default]: [
        { aiType: aiTextureType.DIFFUSE, slot: "BaseColor" },
        { aiType: aiTextureType.SPECULAR, slot: "Specular" },
        { aiType: aiTextureType.EMISSIVE, slot: "Emissive" },
        { aiType: aiTextureType.NORMALS, slot: "Normal" },
    ],
    [ImportMode.OBJ]: [
        { aiType: aiTextureType.DIFFUSE, slot: "BaseColor" },
        { aiType: aiTextureType.SPECULAR, slot: "Specular" },
        { aiType: aiTextureType.EMISSIVE, slot: "Emissive" },
        // OBJ has no normal map, so the bump map stands in for it.
        { aiType: aiTextureType.HEIGHT, slot: "Normal" },
        { aiType: aiTextureType.DISPLACEMENT, slot: "Normal" },
    ],
};

/** Mirrors convertSpecPowerToRoughness (OBJ/MTL Phong exponent -> roughness). */
export function convertSpecPowerToRoughness(specPower: number): number {
    return Math.min(Math.max(Math.sqrt(2 / (specPower + 2)), 0), 1);
}

/**
 * The extensions AssimpImporter registers, minus glTF (its own TS importer),
 * USD and pbrt (separate importers here as natively).
 */
export const kAssimpSceneExtensions = [
    "fbx", "obj", "dae", "x", "md5mesh", "ply", "3ds", "blend", "ase", "ifc", "xgl", "zgl", "dxf", "lwo", "lws",
    "lxo", "stl", "ac", "ms3d", "cob", "scn", "3d", "mdl", "mdl2", "pk3", "smd", "vta", "raw", "ter",
];

export interface AssimpImportOptions {
    assumeLinearSpaceTextures?: boolean;
    /** SceneBuilderFlags::UseSpecGlossMaterials. */
    useSpecGloss?: boolean;
    /** SceneBuilderFlags::UseMetalRoughMaterials. */
    useMetalRough?: boolean;
    /** The scene file's name; its extension picks assimp's loader and the import mode. */
    fileName?: string;
    /** Side files assimp reads next to the scene (an OBJ's `mtllib` files). */
    extraFiles?: { name: string; bytes: Uint8Array }[];
    /** SceneBuilderFlags::DontMergeMeshes (keeps aiProcess_OptimizeMeshes off). */
    dontMergeMeshes?: boolean;
    /** SceneBuilderFlags::UseOriginalTangentSpace (keeps the file's tangents). */
    useOriginalTangentSpace?: boolean;
}

/** `mtllib` references of an OBJ, which assimp resolves by file name. */
export function objMaterialLibraries(source: string): string[] {
    const libs: string[] = [];
    for (const line of source.split(/\r?\n/)) {
        const m = /^\s*mtllib\s+(.+?)\s*$/.exec(line);
        if (m) libs.push(...m[1]!.split(/\s+/));
    }
    return libs;
}

export class FbxImporter {
    /**
     * Parses an assimp-readable scene into scene descs (mirrors
     * AssimpImporter::importInternal). Every format goes through the Default
     * mode except `.obj`, which gets OBJ mode like it does natively.
     */
    static async parseToDescs(
        bytes: Uint8Array,
        baseUrl: string,
        textureManager: TextureManager,
        options: AssimpImportOptions = {},
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; materialNames: string[]; nodes: SceneNode[]; animations: AnimationChannel[]; lights: AnalyticLight[]; cameras: ImportedCamera[] }> {
        if (options.useSpecGloss && options.useMetalRough) {
            throw new RuntimeError("AssimpImporter: UseSpecGlossMaterials and UseMetalRoughMaterials are mutually exclusive");
        }
        const fileName = options.fileName ?? "scene.fbx";
        const importMode = fileName.toLowerCase().endsWith(".obj") ? ImportMode.OBJ : ImportMode.Default;
        // MetalRough everywhere except OBJ, unless a flag says otherwise.
        const shadingModel =
            options.useSpecGloss || (importMode === ImportMode.OBJ && !options.useMetalRough) ? ShadingModel.SpecGloss : ShadingModel.MetalRough;

        // AssimpImporter::importInternal's flags and component removal.
        let flags = kTargetRealtimeMaxQuality | aiProcess.FlipUVs | aiProcess.RemoveComponent;
        flags &= ~(aiProcess.CalcTangentSpace | aiProcess.FindDegenerates | aiProcess.OptimizeGraph | aiProcess.RemoveRedundantMaterials | aiProcess.SplitLargeMeshes);
        if (options.dontMergeMeshes) flags &= ~aiProcess.OptimizeMeshes;
        // aiComponent_COLORS | aiComponent_TEXCOORDSn(1..7) (n = 7 shifts out of 32 bits), plus tangents.
        let removeComponents = 0x8;
        for (let layer = 1; layer < 7; layer++) removeComponents |= 1 << (layer + 25);
        if (!options.useOriginalTangentSpace) removeComponents |= 0x4;
        const json = await assimpImport([{ name: fileName, bytes }, ...(options.extraFiles ?? [])], flags, removeComponents, "FbxImporter");

        // Textures (loaded per unique path; slot decides sRGB like loadMaterialTexture).
        const textureIDs = new Map<string, number>();
        const skippedFormats = new Set<string>();
        type DecodedTexture = Parameters<TextureManager["addTexture"]>[0];
        // Fetch + decode run concurrently (a few at a time); registration below stays in load order,
        // so texture IDs match a sequential load.
        const decodes = new Map<string, Promise<DecodedTexture | undefined>>();
        let inFlight = 0;
        const waiting: (() => void)[] = [];
        const limited = async <T>(job: () => Promise<T>): Promise<T> => {
            if (inFlight >= 8) await new Promise<void>((resolve) => waiting.push(resolve));
            inFlight++;
            try {
                return await job();
            } finally {
                inFlight--;
                waiting.shift()?.();
            }
        };
        const textureKey = (path: string, slotSrgb: boolean) => {
            const srgb = slotSrgb && !options.assumeLinearSpaceTextures;
            const norm = path.replace(/\\/g, "/");
            return { norm, srgb, key: `${norm}|${srgb}` };
        };
        const decodeTexture = (path: string, slotSrgb: boolean): Promise<DecodedTexture | undefined> => {
            const { norm, srgb, key } = textureKey(path, slotSrgb);
            let pending = decodes.get(key);
            if (!pending) {
                pending = limited(() => fetchAndDecode(norm, srgb));
                decodes.set(key, pending);
            }
            return pending;
        };
        const loadTexture = async (path: string, slotSrgb: boolean): Promise<number | undefined> => {
            const { key } = textureKey(path, slotSrgb);
            if (textureIDs.has(key)) return textureIDs.get(key);
            const decoded = await decodeTexture(path, slotSrgb);
            if (!decoded) return undefined;
            const id = textureManager.addTexture(decoded);
            textureIDs.set(key, id);
            return id;
        };
        const fetchAndDecode = async (norm: string, srgb: boolean): Promise<DecodedTexture | undefined> => {
            const url = baseUrl ? `${baseUrl}/${norm}` : norm;
            const res = await fetch(url);
            if (!res.ok) return undefined;
            const ext = norm.slice(norm.lastIndexOf(".")).toLowerCase();
            // The browser's createImageBitmap decodes png/jpg/webp/bmp only.
            // BC-compressed DDS (the common game-asset format — Bistro, Sponza,
            // SunTemple) is decoded here on the CPU to RGBA8 at a bounded size
            // (decodeDDSToRGBA caps the mip) so it feeds the existing RGBA8
            // texture-array path, and TGA goes through the CPU decoder. Other
            // undecodable formats skip so geometry loads with a base colour.
            let bitmap: ImageBitmap;
            let compressed: ReturnType<typeof ddsCompressedPayload>;
            let ddsBytes: Uint8Array | undefined;
            try {
                if (ext === ".tga") {
                    // Browsers cannot decode TGA; native reads it through FreeImage.
                    const buffer = await res.arrayBuffer();
                    const image = await WorkerPool.get().run("decodeTGA", { buffer }, [buffer]);
                    bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(image.rgba), image.width, image.height), {
                        premultiplyAlpha: "none",
                        colorSpaceConversion: "none",
                    });
                } else if (ext === ".dds") {
                    const buffer = await res.arrayBuffer();
                    // The GPU gets the full-resolution BC chain; the capped decode serves CPU analysis.
                    compressed = ddsCompressedPayload(buffer, srgb);
                    ddsBytes = new Uint8Array(buffer);
                    const copy = buffer.slice(0);
                    const { width, height, rgba } = await WorkerPool.get().run("decodeDDS", { buffer: copy, srgb, maxDim: 512 }, [copy]);
                    // ImageData holds raw RGBA already — no colour-space/premultiply
                    // decode step applies, so createImageBitmap needs no options.
                    const imageData = new ImageData(new Uint8ClampedArray(rgba), width, height);
                    bitmap = await createImageBitmap(imageData);
                } else {
                    bitmap = await createImageBitmap(await res.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
                }
            } catch {
                skippedFormats.add(ext);
                return undefined;
            }
            return { bitmap, srgb, compressed, bytes: ddsBytes, dds: ddsBytes !== undefined };
        };

        // Start every material texture's fetch + decode up front.
        for (const mat of json.materials) {
            for (const { aiType, slot } of kTextureMappings[importMode]) {
                const file = textureFile(mat, aiType);
                if (file) void decodeTexture(file, getTextureSlotSrgb(MaterialType.Standard, shadingModel, slot) ?? false);
            }
        }

        // Materials (createMaterial).
        const materials: SceneMaterialDesc[] = [];
        const materialNames: string[] = [];
        for (const mat of json.materials) {
            const name = String(findProp(mat, "?mat.name") ?? "unnamed");
            materialNames.push(name);

            const diffuse = (findProp(mat, "$clr.diffuse") as number[] | undefined) ?? [1, 1, 1];
            const specular = (findProp(mat, "$clr.specular") as number[] | undefined) ?? [0, 0, 0];
            const emissive = (findProp(mat, "$clr.emissive") as number[] | undefined) ?? [0, 0, 0];
            const opacity = (findProp(mat, "$mat.opacity") as number | undefined) ?? 1;
            const shininessProp = findProp(mat, "$mat.shininess") as number | undefined;
            // OBJ/MTL carries a Phong exponent; native converts it to glossiness.
            const shininess =
                shininessProp === undefined ? 0 : importMode === ImportMode.OBJ ? 1 - convertSpecPowerToRoughness(shininessProp) : shininessProp;
            const refracti = findProp(mat, "$mat.refracti") as number | undefined;
            const twosided = findProp(mat, "$mat.twosided") as number | undefined;

            // Name suffix flags (tokens after '.').
            let doubleSided = twosided !== undefined && twosided !== 0;
            for (const token of name.split(".").slice(1)) {
                if (token.toLowerCase() === "doublesided") doubleSided = true;
            }

            // Texture table per import mode; each slot's colour space comes from
            // the StandardMaterial slot table for this shading model.
            const slotFiles: Partial<Record<"BaseColor" | "Specular" | "Emissive" | "Normal", string>> = {};
            for (const { aiType, slot } of kTextureMappings[importMode]) {
                const file = textureFile(mat, aiType);
                if (file) slotFiles[slot] = file;
            }
            const load = async (slot: "BaseColor" | "Specular" | "Emissive" | "Normal") => {
                const file = slotFiles[slot];
                return file === undefined ? undefined : loadTexture(file, getTextureSlotSrgb(MaterialType.Standard, shadingModel, slot) ?? false);
            };
            const ids = {
                baseColor: await load("BaseColor"),
                specular: await load("Specular"),
                emissive: await load("Emissive"),
                normal: await load("Normal"),
            };

            materials.push({
                header: {
                    doubleSided,
                    emissive: emissive.some((c) => c !== 0) || ids.emissive !== undefined,
                },
                basic: {
                    baseColor: new float4(diffuse[0]!, diffuse[1]!, diffuse[2]!, opacity),
                    // Native writes COLOR_SPECULAR into rgb and raw shininess into a.
                    specular: new float4(specular[0]!, specular[1]!, specular[2]!, shininess),
                    emissive: new float3(emissive[0]!, emissive[1]!, emissive[2]!),
                    shadingModel,
                    ...(refracti !== undefined ? { indexOfRefraction: refracti } : {}),
                    ...(opacity < 1 ? { specularTransmission: 1 - opacity } : {}),
                    texBaseColor: ids.baseColor !== undefined ? packTextureHandle(TextureHandleMode.Texture, ids.baseColor) : undefined,
                    texSpecular: ids.specular !== undefined ? packTextureHandle(TextureHandleMode.Texture, ids.specular) : undefined,
                    texEmissive: ids.emissive !== undefined ? packTextureHandle(TextureHandleMode.Texture, ids.emissive) : undefined,
                    texNormalMap: ids.normal !== undefined ? packTextureHandle(TextureHandleMode.Texture, ids.normal) : undefined,
                },
            });
        }

        // Node hierarchy -> mesh instances (aiMatrix4x4 is row-major).
        const meshDescs: SceneMeshDesc[] = [];
        const meshVertices = new Map<number, { vertices: StaticVertex[]; indices: Uint32Array }>();
        const getMesh = (mi: number) => {
            let cached = meshVertices.get(mi);
            if (!cached) {
                const mesh = json.meshes[mi]!;
                const count = mesh.vertices.length / 3;
                const uvs = mesh.texturecoords?.[0];
                const uvStride = mesh.numuvcomponents?.[0] ?? 2;
                const vertices: StaticVertex[] = [];
                for (let i = 0; i < count; i++) {
                    vertices.push({
                        position: new float3(mesh.vertices[i * 3]!, mesh.vertices[i * 3 + 1]!, mesh.vertices[i * 3 + 2]!),
                        normal: mesh.normals
                            ? new float3(mesh.normals[i * 3]!, mesh.normals[i * 3 + 1]!, mesh.normals[i * 3 + 2]!)
                            : new float3(0, 0, 1),
                        tangent: new float4(0, 0, 0, 0),
                        // aiProcess_FlipUVs already flipped V.
                        texCrd: uvs ? new float2(uvs[i * uvStride]!, uvs[i * uvStride + 1]!) : new float2(0, 0),
                    });
                }
                const idx = new Uint32Array(mesh.triangles);
                cached = { vertices, indices: idx };
                meshVertices.set(mi, cached);
            }
            return cached;
        };

        // Retained node graph for animation (assimp channels target nodes by name).
        const nodes: SceneNode[] = [];
        const nameToNodeID = new Map<string, number>();
        const nodeNames: string[] = [];
        const nameToWorld = new Map<string, float4x4>(); // for placing lights on their nodes
        const visit = (node: AiNode, parentWorld: float4x4, parentID: number) => {
            const local = new float4x4();
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) local.set(r, c, node.transformation[r * 4 + c]!);
            const world = mulMat(parentWorld, local);
            const nodeID = nodes.length;
            nodes.push({ parent: parentID, ...decomposeTRS(local) });
            nodeNames[nodeID] = node.name;
            if (node.name) {
                nameToNodeID.set(node.name, nodeID);
                nameToWorld.set(node.name, world);
            }
            for (const mi of node.meshes ?? []) {
                const { vertices, indices } = getMesh(mi);
                meshDescs.push({ vertices, indices, materialID: json.meshes[mi]!.materialindex, transform: world, nodeID, tangentSpace: "generate" });
                skinnedDescs.push({ desc: meshDescs[meshDescs.length - 1]!, mi });
            }
            for (const child of node.children ?? []) visit(child, world, nodeID);
        };
        // Deferred skin attach: bone->node lookup needs the full node graph first.
        const skinnedDescs: { desc: SceneMeshDesc; mi: number }[] = [];
        visit(json.rootnode, float4x4.identity(), -1);
        if (meshDescs.length === 0) throw new RuntimeError("FbxImporter: no triangle meshes found");

        // Skinning: assjson bones carry a node name, mesh-space inverse-bind
        // (offsetmatrix, row-major) and [vertexId, weight] lists. Build one
        // SkinDesc per skinned mesh (top-4 influences per vertex, renormalized)
        // — the same CPU linear-blend path glTF skins use in Scene.animate().
        const skinCache = new Map<number, SkinDesc | undefined>();
        const buildSkin = (mi: number): SkinDesc | undefined => {
            if (skinCache.has(mi)) return skinCache.get(mi);
            const bones = json.meshes[mi]!.bones;
            const vertCount = json.meshes[mi]!.vertices.length / 3;
            let skin: SkinDesc | undefined;
            if (bones && bones.length > 0) {
                const influences: { bone: number; weight: number }[][] = Array.from({ length: vertCount }, () => []);
                bones.forEach((bone, bi) => {
                    for (const [vid, w] of bone.weights) if (w > 0 && influences[vid]) influences[vid]!.push({ bone: bi, weight: w });
                });
                const boneIDs = new Uint32Array(vertCount * 4);
                const weights = new Float32Array(vertCount * 4);
                for (let v = 0; v < vertCount; v++) {
                    const inf = influences[v]!.sort((a, b) => b.weight - a.weight).slice(0, 4);
                    const sum = inf.reduce((s, x) => s + x.weight, 0) || 1;
                    for (let k = 0; k < inf.length; k++) {
                        boneIDs[v * 4 + k] = inf[k]!.bone;
                        weights[v * 4 + k] = inf[k]!.weight / sum;
                    }
                }
                skin = {
                    boneNodeIDs: bones.map((b) => nameToNodeID.get(b.name) ?? 0),
                    inverseBind: bones.map((b) => {
                        const m = new float4x4();
                        for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m.set(r, c, b.offsetmatrix[r * 4 + c]!);
                        return m;
                    }),
                    boneIDs,
                    weights,
                };
            }
            skinCache.set(mi, skin);
            return skin;
        };
        for (const { desc, mi } of skinnedDescs) {
            const skin = buildSkin(mi);
            if (skin) desc.skin = skin;
        }

        // Animation channels (assimp: per-node position/rotation/scaling key tracks;
        // times in ticks -> seconds; rotation quaternions are [w,x,y,z]).
        const animations: AnimationChannel[] = [];
        // Clip ordinal mirrors native (one Animation per assimp node-anim, in
        // order) so pyscene `sceneBuilder.animations[i]` behavior writes land.
        let clip = 0;
        for (const anim of json.animations ?? []) {
            const tps = anim.tickspersecond && anim.tickspersecond > 0 ? anim.tickspersecond : 24;
            for (const ch of anim.channels ?? []) {
                const thisClip = clip++; // count every node-anim (native creates an Animation even for unmatched nodes)
                const nodeID = nameToNodeID.get(ch.name);
                if (nodeID === undefined) continue;
                const track = (keys: AiKey[] | undefined, path: "translation" | "rotation" | "scale", quat: boolean) => {
                    if (!keys?.length) return;
                    const times = new Float32Array(keys.map((k) => k[0] / tps));
                    const values = new Float32Array(quat ? keys.flatMap((k) => [k[1][1]!, k[1][2]!, k[1][3]!, k[1][0]!]) : keys.flatMap((k) => k[1]));
                    animations.push({ nodeID, path, times, values, interp: "LINEAR", clip: thisClip });
                };
                track(ch.positionkeys, "translation", false);
                track(ch.rotationkeys, "rotation", true);
                track(ch.scalingkeys, "scale", false);
            }
        }

        // Cameras (createCameras): the pose is the aiCamera's local one; an animated camera node
        // drives it through a local child node holding the camera's view matrix.
        const cameras: ImportedCamera[] = [];
        for (const C of json.cameras ?? []) {
            const v = (a: number[] | undefined, d: float3) => (a ? new float3(a[0]!, a[1]!, a[2]!) : d);
            const position = v(C.position, new float3(0, 0, 0));
            const look = v(C.lookat, new float3(0, 0, -1));
            const cam = new Camera(C.name);
            cam.setPosition(position);
            cam.setUpVector(v(C.up, new float3(0, 1, 0)));
            cam.setTarget(new float3(look.x + position.x, look.y + position.y, look.z + position.z));
            const imported: ImportedCamera = {
                name: C.name,
                // FBX keeps a fixed 35mm focal length (native backwards compatibility).
                pose: { position, target: cam.getTarget(), up: cam.getUpVector(), focalLength: 35, depthRange: [C.clipplanenear ?? 0.1, C.clipplanefar ?? 1000] },
                aspectRatio: C.aspect || undefined,
            };
            const nodeID = nameToNodeID.get(C.name);
            const nodeWorld = nameToWorld.get(C.name);
            if (nodeWorld && !(nodeID !== undefined && animations.some((a) => a.nodeID === nodeID))) {
                // Static cameras too take their node's transform (Camera::updateFromAnimation of
                // the node world * the camera's local view-matrix node): up, -forward, position columns.
                const g = mulMat(nodeWorld, cam.getViewMatrix());
                const pos = new float3(g.get(0, 3), g.get(1, 3), g.get(2, 3));
                imported.pose.position = pos;
                imported.pose.up = new float3(g.get(0, 1), g.get(1, 1), g.get(2, 1));
                imported.pose.target = new float3(pos.x - g.get(0, 2), pos.y - g.get(1, 2), pos.z - g.get(2, 2));
            }
            if (nodeID !== undefined && animations.some((a) => a.nodeID === nodeID)) {
                imported.nodeID = nodes.length;
                nodes.push({ parent: nodeID, ...decomposeTRS(cam.getViewMatrix()) });
                // fixFbxCameraAnimation: the animation already holds the pivot helpers' transforms.
                const prefix = `${C.name}_$AssimpFbx$_`;
                for (let p = nodes[nodeID]!.parent; p >= 0 && nodeNames[p]?.startsWith(prefix); p = nodes[p]!.parent) Object.assign(nodes[p]!, decomposeTRS(float4x4.identity()));
            }
            cameras.push(imported);
        }

        // Analytic lights (directional/point), placed by their node's world transform.
        const lights: AnalyticLight[] = [];
        for (const L of json.lights ?? []) {
            const nodeWorld = nameToWorld.get(L.name) ?? float4x4.identity();
            const c = L.diffusecolor ?? [1, 1, 1];
            const intensity = new float3(c[0]!, c[1]!, c[2]!);
            if (L.type === 1 && L.direction) {
                lights.push({ type: LightType.Directional, dirW: normalize3(transformVector(nodeWorld, new float3(L.direction[0]!, L.direction[1]!, L.direction[2]!))), intensity });
            } else if (L.type === 2) {
                lights.push({ type: LightType.Point, posW: transformPoint(nodeWorld, new float3(0, 0, 0)), intensity });
            }
        }

        if (skippedFormats.size > 0) {
            console.warn(`FbxImporter: skipped textures with undecodable formats [${[...skippedFormats].join(", ")}] (no decoder for them); materials fall back to base color.`);
        }
        // Natively a material enters the scene only when a mesh adds it, in
        // first-use order; assimp's unused ones (OBJ's DefaultMaterial) never do.
        const remap = new Map<number, number>();
        const usedMaterials: SceneMaterialDesc[] = [];
        const usedNames: string[] = [];
        for (const mesh of meshDescs) {
            let id = remap.get(mesh.materialID);
            if (id === undefined) {
                id = usedMaterials.length;
                remap.set(mesh.materialID, id);
                usedMaterials.push(materials[mesh.materialID]!);
                usedNames.push(materialNames[mesh.materialID]!);
            }
            mesh.materialID = id;
        }
        return { meshes: meshDescs, materials: usedMaterials, materialNames: usedNames, nodes, animations, lights, cameras };
    }

    /** Parses a single mesh asset (.obj/.ply/etc. via assimp) into one merged
     *  TriangleMesh, for TriangleMesh.createFromFile(). Mirrors native
     *  TriangleMesh::createFromFile postprocessing: node transforms baked
     *  (aiProcess_PreTransformVertices), V flipped (aiProcess_FlipUVs), and
     *  missing normals generated — flat per-face by default, smooth when
     *  smoothNormals is set (aiProcess_GenNormals / GenSmoothNormals).
     *  Materials are ignored (the caller assigns its own material/instance).
     *  `filename` must keep the real extension so assimp picks the right importer. */
    static async parseMeshOnly(bytes: Uint8Array, filename: string, smoothNormals = false): Promise<{ vertices: StaticVertex[]; indices: Uint32Array }> {
        // TriangleMesh::createFromFile's flags (default ImportFlags: no JoinIdenticalVertices).
        const flags = aiProcess.FlipUVs | aiProcess.Triangulate | aiProcess.PreTransformVertices | (smoothNormals ? aiProcess.GenSmoothNormals : aiProcess.GenNormals);
        const json = await assimpImport([{ name: filename.split("/").pop()!, bytes }], flags, 0, `TriangleMesh.createFromFile('${filename}')`);

        // Bake node transforms (native aiProcess_PreTransformVertices).
        const meshWorld = new Map<number, float4x4>();
        const visit = (node: AiNode, parentWorld: float4x4) => {
            const local = new float4x4();
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) local.set(r, c, node.transformation[r * 4 + c]!);
            const world = mulMat(parentWorld, local);
            for (const mi of node.meshes ?? []) meshWorld.set(mi, world);
            for (const child of node.children ?? []) visit(child, world);
        };
        if (json.rootnode) visit(json.rootnode, float4x4.identity());

        const vertices: StaticVertex[] = [];
        const indices: number[] = [];
        (json.meshes ?? []).forEach((mesh, mi) => {
            const base = vertices.length;
            const count = mesh.vertices.length / 3;
            const uvs = mesh.texturecoords?.[0];
            const uvStride = mesh.numuvcomponents?.[0] ?? 2;
            const world = meshWorld.get(mi) ?? float4x4.identity();
            for (let i = 0; i < count; i++) {
                const p = transformPoint(world, new float3(mesh.vertices[i * 3]!, mesh.vertices[i * 3 + 1]!, mesh.vertices[i * 3 + 2]!));
                const n = mesh.normals
                    ? normalize3(transformVector(world, new float3(mesh.normals[i * 3]!, mesh.normals[i * 3 + 1]!, mesh.normals[i * 3 + 2]!)))
                    : new float3(0, 0, 1); // regenerated below when absent
                vertices.push({
                    position: p,
                    normal: n,
                    tangent: new float4(0, 0, 0, 0),
                    // aiProcess_FlipUVs already flipped V.
                    texCrd: uvs ? new float2(uvs[i * uvStride]!, uvs[i * uvStride + 1]!) : new float2(0, 0),
                });
            }
            for (const i of mesh.triangles) indices.push(base + i);
        });
        if (vertices.length === 0) throw new RuntimeError(`TriangleMesh.createFromFile('${filename}'): no geometry`);

        // aiProcess_GenNormals / GenSmoothNormals normally leave none missing; kept as a fallback.
        const missingNormals = (json.meshes ?? []).some((m) => !m.normals);
        let idx = new Uint32Array(indices);
        if (missingNormals && !smoothNormals) {
            // Flat per-face normals (aiProcess_GenNormals): split vertices per face.
            const flat: StaticVertex[] = [];
            for (let f = 0; f < idx.length; f += 3) {
                const [a, b, c] = [vertices[idx[f]!]!, vertices[idx[f + 1]!]!, vertices[idx[f + 2]!]!];
                const n = normalize3(cross(sub3(b.position, a.position), sub3(c.position, a.position)));
                for (const v of [a, b, c]) flat.push({ ...v, normal: n });
            }
            vertices.length = 0;
            vertices.push(...flat);
            idx = new Uint32Array(flat.length);
            for (let i = 0; i < flat.length; i++) idx[i] = i;
        } else if (missingNormals) {
            // Smooth normals (aiProcess_GenSmoothNormals): area-weighted average
            // over the faces sharing each vertex (assimp already joined
            // identical vertices).
            const acc = vertices.map(() => new float3(0, 0, 0));
            for (let f = 0; f < idx.length; f += 3) {
                const [a, b, c] = [idx[f]!, idx[f + 1]!, idx[f + 2]!];
                const n = cross(sub3(vertices[b]!.position, vertices[a]!.position), sub3(vertices[c]!.position, vertices[a]!.position));
                for (const vi of [a, b, c]) acc[vi] = add3(acc[vi]!, n);
            }
            vertices.forEach((v, i) => (v.normal = normalize3(acc[i]!)));
        }
        // Tangents come from SceneBuilder, as for any TriangleMesh.
        return { vertices, indices: idx };
    }
}
