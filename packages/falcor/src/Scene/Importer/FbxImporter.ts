/**
 * FBX importer mirroring plugins/importers/AssimpImporter (Default import
 * mode). Parsing runs through assimpjs (the same Assimp library compiled to
 * WASM, npm package) emitting the aiScene as JSON; this module ports the
 * native mapping: node-hierarchy flattening, Default-mode material semantics
 * (diffuse/specular/emissive colors, shininess into specular.a, opacity ->
 * specular transmission, ".DoubleSided" name suffix) and the Default-mode
 * texture-slot table (DIFFUSE->BaseColor, SPECULAR->Specular,
 * EMISSIVE->Emissive, NORMALS->Normal).
 *
 * Divergence (documented): native runs Assimp with
 * aiProcessPreset_TargetRealtime_MaxQuality; assimpjs uses its own fixed
 * post-process flags, so vertex counts may differ (JoinIdenticalVertices) —
 * geometry is verified against native renders instead of buffer equality.
 */

import { float2, float3, float4, normalize3, cross, sub3, add3 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { generateTangents } from "../TangentSpace.js";
import { MaterialType, ShadingModel, packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import { getTextureSlotSrgb } from "../Material/TextureSlots.js";
import { decodeTGA } from "../../Utils/Image/TGADecoder.js";
import { decodeDDSToRGBA } from "./DDSLoader.js";
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
    vertices: number[];
    normals?: number[];
    texturecoords?: number[][];
    faces: number[][];
    bones?: AiBone[];
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

let assimpModule: unknown | null = null;

interface AssimpApi {
    FileList: new () => { AddFile(name: string, data: Uint8Array): void };
    ConvertFileList(
        files: unknown,
        format: string,
    ): { IsSuccess(): boolean; GetErrorCode(): string; FileCount(): number; GetFile(i: number): { GetContent(): Uint8Array } };
}

/** Loads the assimpjs WASM module (emscripten UMD script from node_modules). */
async function getAssimp(): Promise<AssimpApi> {
    if (!assimpModule) {
        const g = globalThis as { assimpjs?: (opts?: object) => Promise<unknown> };
        if (!g.assimpjs) {
            await new Promise<void>((resolveScript, reject) => {
                const script = document.createElement("script");
                script.src = "/node_modules/assimpjs/dist/assimpjs.js";
                script.onload = () => resolveScript();
                script.onerror = () => reject(new RuntimeError("FbxImporter: failed to load assimpjs"));
                document.head.appendChild(script);
            });
        }
        assimpModule = await g.assimpjs!({
            locateFile: (file: string) => `/node_modules/assimpjs/dist/${file}`,
        });
    }
    return assimpModule as AssimpApi;
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

        const ajs = await getAssimp();
        const files = new ajs.FileList();
        files.AddFile(fileName, bytes);
        for (const extra of options.extraFiles ?? []) files.AddFile(extra.name, extra.bytes);
        const result = ajs.ConvertFileList(files, "assjson");
        if (!result.IsSuccess()) throw new RuntimeError(`FbxImporter: assimp failed (${result.GetErrorCode()})`);
        const json = JSON.parse(new TextDecoder().decode(result.GetFile(0).GetContent())) as AiScene;

        // Textures (loaded per unique path; slot decides sRGB like loadMaterialTexture).
        const textureIDs = new Map<string, number>();
        const skippedFormats = new Set<string>();
        const loadTexture = async (path: string, slotSrgb: boolean): Promise<number | undefined> => {
            const srgb = slotSrgb && !options.assumeLinearSpaceTextures;
            const norm = path.replace(/\\/g, "/");
            const key = `${norm}|${srgb}`;
            if (textureIDs.has(key)) return textureIDs.get(key);
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
            try {
                if (ext === ".tga") {
                    // Browsers cannot decode TGA; native reads it through FreeImage.
                    const image = decodeTGA(await res.arrayBuffer());
                    bitmap = await createImageBitmap(new ImageData(new Uint8ClampedArray(image.rgba), image.width, image.height), {
                        premultiplyAlpha: "none",
                        colorSpaceConversion: "none",
                    });
                } else if (ext === ".dds") {
                    const { width, height, rgba } = decodeDDSToRGBA(await res.arrayBuffer(), srgb, 512);
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
            const id = textureManager.addTexture({ bitmap, srgb });
            textureIDs.set(key, id);
            return id;
        };

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
                const vertices: StaticVertex[] = [];
                for (let i = 0; i < count; i++) {
                    vertices.push({
                        position: new float3(mesh.vertices[i * 3]!, mesh.vertices[i * 3 + 1]!, mesh.vertices[i * 3 + 2]!),
                        normal: mesh.normals
                            ? new float3(mesh.normals[i * 3]!, mesh.normals[i * 3 + 1]!, mesh.normals[i * 3 + 2]!)
                            : new float3(0, 0, 1),
                        tangent: new float4(0, 0, 0, 0),
                        // Native imports with aiProcess_FlipUVs; assimpjs does not
                        // flip (verified vs the Arcade oracle: unflipped is worse).
                        texCrd: uvs ? new float2(uvs[i * 2]!, 1 - uvs[i * 2 + 1]!) : new float2(0, 0),
                    });
                }
                const indices: number[] = [];
                for (const face of mesh.faces) {
                    if (face.length === 3) indices.push(face[0]!, face[1]!, face[2]!);
                }
                const idx = new Uint32Array(indices);
                generateTangents(vertices, idx);
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
                meshDescs.push({ vertices, indices, materialID: json.meshes[mi]!.materialindex, transform: world, nodeID });
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
        const ajs = await getAssimp();
        const files = new ajs.FileList();
        files.AddFile(filename, bytes);
        const result = ajs.ConvertFileList(files, "assjson");
        if (!result.IsSuccess()) throw new RuntimeError(`TriangleMesh.createFromFile('${filename}'): assimp failed (${result.GetErrorCode()})`);
        const json = JSON.parse(new TextDecoder().decode(result.GetFile(0).GetContent())) as AiScene;

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
                    // Native imports with aiProcess_FlipUVs.
                    texCrd: uvs ? new float2(uvs[i * 2]!, 1 - uvs[i * 2 + 1]!) : new float2(0, 0),
                });
            }
            for (const face of mesh.faces) if (face.length === 3) indices.push(base + face[0]!, base + face[1]!, base + face[2]!);
        });
        if (vertices.length === 0) throw new RuntimeError(`TriangleMesh.createFromFile('${filename}'): no geometry`);

        // Generate normals when the asset has none (assimpjs runs no GenNormals pass).
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
            // over the faces sharing each vertex (assimpjs already joined
            // identical vertices).
            const acc = vertices.map(() => new float3(0, 0, 0));
            for (let f = 0; f < idx.length; f += 3) {
                const [a, b, c] = [idx[f]!, idx[f + 1]!, idx[f + 2]!];
                const n = cross(sub3(vertices[b]!.position, vertices[a]!.position), sub3(vertices[c]!.position, vertices[a]!.position));
                for (const vi of [a, b, c]) acc[vi] = add3(acc[vi]!, n);
            }
            vertices.forEach((v, i) => (v.normal = normalize3(acc[i]!)));
        }
        generateTangents(vertices, idx);
        return { vertices, indices: idx };
    }
}
