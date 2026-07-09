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

import { float2, float3, float4, normalize3 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { generateTangents } from "../TangentSpace.js";
import { packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import { decodeDDSToRGBA } from "./DDSLoader.js";
import type { SceneMaterialDesc, SceneMeshDesc } from "../Scene.js";
import { decomposeTRS, type SceneNode, type AnimationChannel, type SkinDesc } from "../Animation/SceneAnimation.js";
import { LightType, type AnalyticLight, type StaticVertex } from "../SceneData.js";
import type { TextureManager } from "../Material/TextureManager.js";

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

export class FbxImporter {
    /** Parses an .fbx buffer into scene descs (mirrors AssimpImporter::importInternal, Default mode). */
    static async parseToDescs(
        bytes: Uint8Array,
        baseUrl: string,
        textureManager: TextureManager,
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; materialNames: string[]; nodes: SceneNode[]; animations: AnimationChannel[]; lights: AnalyticLight[] }> {
        const ajs = await getAssimp();
        const files = new ajs.FileList();
        files.AddFile("scene.fbx", bytes);
        const result = ajs.ConvertFileList(files, "assjson");
        if (!result.IsSuccess()) throw new RuntimeError(`FbxImporter: assimp failed (${result.GetErrorCode()})`);
        const json = JSON.parse(new TextDecoder().decode(result.GetFile(0).GetContent())) as AiScene;

        // Textures (loaded per unique path; slot decides sRGB like loadMaterialTexture).
        const textureIDs = new Map<string, number>();
        const skippedFormats = new Set<string>();
        const loadTexture = async (path: string, srgb: boolean): Promise<number | undefined> => {
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
            // texture-array path. Other undecodable formats (e.g. TGA) still
            // skip gracefully so geometry loads with a base-colour fallback.
            let bitmap: ImageBitmap;
            try {
                if (ext === ".dds") {
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

        // Materials (createMaterial, Default mode: shading model MetalRough).
        const materials: SceneMaterialDesc[] = [];
        const materialNames: string[] = [];
        for (const mat of json.materials) {
            const name = String(findProp(mat, "?mat.name") ?? "unnamed");
            materialNames.push(name);

            const diffuse = (findProp(mat, "$clr.diffuse") as number[] | undefined) ?? [1, 1, 1];
            const specular = (findProp(mat, "$clr.specular") as number[] | undefined) ?? [0, 0, 0];
            const emissive = (findProp(mat, "$clr.emissive") as number[] | undefined) ?? [0, 0, 0];
            const opacity = (findProp(mat, "$mat.opacity") as number | undefined) ?? 1;
            const shininess = (findProp(mat, "$mat.shininess") as number | undefined) ?? 0;
            const refracti = findProp(mat, "$mat.refracti") as number | undefined;
            const twosided = findProp(mat, "$mat.twosided") as number | undefined;

            // Name suffix flags (tokens after '.').
            let doubleSided = twosided !== undefined && twosided !== 0;
            for (const token of name.split(".").slice(1)) {
                if (token.toLowerCase() === "doublesided") doubleSided = true;
            }

            const texBaseColor = textureFile(mat, 1);
            const texSpecular = textureFile(mat, 2);
            const texEmissive = textureFile(mat, 4);
            const texNormal = textureFile(mat, 6);

            const ids = {
                baseColor: texBaseColor !== undefined ? await loadTexture(texBaseColor, true) : undefined,
                specular: texSpecular !== undefined ? await loadTexture(texSpecular, false) : undefined,
                emissive: texEmissive !== undefined ? await loadTexture(texEmissive, true) : undefined,
                normal: texNormal !== undefined ? await loadTexture(texNormal, false) : undefined,
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
        const nameToWorld = new Map<string, float4x4>(); // for placing lights on their nodes
        const visit = (node: AiNode, parentWorld: float4x4, parentID: number) => {
            const local = new float4x4();
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) local.set(r, c, node.transformation[r * 4 + c]!);
            const world = mulMat(parentWorld, local);
            const nodeID = nodes.length;
            nodes.push({ parent: parentID, ...decomposeTRS(local) });
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
        for (const anim of json.animations ?? []) {
            const tps = anim.tickspersecond && anim.tickspersecond > 0 ? anim.tickspersecond : 24;
            for (const ch of anim.channels ?? []) {
                const nodeID = nameToNodeID.get(ch.name);
                if (nodeID === undefined) continue;
                const track = (keys: AiKey[] | undefined, path: "translation" | "rotation" | "scale", quat: boolean) => {
                    if (!keys?.length) return;
                    const times = new Float32Array(keys.map((k) => k[0] / tps));
                    const values = new Float32Array(quat ? keys.flatMap((k) => [k[1][1]!, k[1][2]!, k[1][3]!, k[1][0]!]) : keys.flatMap((k) => k[1]));
                    animations.push({ nodeID, path, times, values, interp: "LINEAR" });
                };
                track(ch.positionkeys, "translation", false);
                track(ch.rotationkeys, "rotation", true);
                track(ch.scalingkeys, "scale", false);
            }
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
            console.warn(`FbxImporter: skipped textures with undecodable formats [${[...skippedFormats].join(", ")}] (need a DDS/BC or TGA decoder); materials fall back to base color.`);
        }
        return { meshes: meshDescs, materials, materialNames, nodes, animations, lights };
    }

    /** Parses a single mesh asset (.obj/.ply/etc. via assimp) into one merged
     *  local-space TriangleMesh, for TriangleMesh.createFromFile(). Materials and
     *  node transforms are ignored (the caller assigns its own material/instance).
     *  `filename` must keep the real extension so assimp picks the right importer. */
    static async parseMeshOnly(bytes: Uint8Array, filename: string): Promise<{ vertices: StaticVertex[]; indices: Uint32Array }> {
        const ajs = await getAssimp();
        const files = new ajs.FileList();
        files.AddFile(filename, bytes);
        const result = ajs.ConvertFileList(files, "assjson");
        if (!result.IsSuccess()) throw new RuntimeError(`TriangleMesh.createFromFile('${filename}'): assimp failed (${result.GetErrorCode()})`);
        const json = JSON.parse(new TextDecoder().decode(result.GetFile(0).GetContent())) as AiScene;
        const vertices: StaticVertex[] = [];
        const indices: number[] = [];
        for (const mesh of json.meshes ?? []) {
            const base = vertices.length;
            const count = mesh.vertices.length / 3;
            const uvs = mesh.texturecoords?.[0];
            for (let i = 0; i < count; i++) {
                vertices.push({
                    position: new float3(mesh.vertices[i * 3]!, mesh.vertices[i * 3 + 1]!, mesh.vertices[i * 3 + 2]!),
                    normal: mesh.normals ? new float3(mesh.normals[i * 3]!, mesh.normals[i * 3 + 1]!, mesh.normals[i * 3 + 2]!) : new float3(0, 0, 1),
                    tangent: new float4(0, 0, 0, 0),
                    texCrd: uvs ? new float2(uvs[i * 2]!, uvs[i * 2 + 1]!) : new float2(0, 0),
                });
            }
            for (const face of mesh.faces) if (face.length === 3) indices.push(base + face[0]!, base + face[1]!, base + face[2]!);
        }
        if (vertices.length === 0) throw new RuntimeError(`TriangleMesh.createFromFile('${filename}'): no geometry`);
        const idx = new Uint32Array(indices);
        generateTangents(vertices, idx);
        return { vertices, indices: idx };
    }
}
