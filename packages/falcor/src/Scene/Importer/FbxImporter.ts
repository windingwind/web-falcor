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

import { float2, float3, float4 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { generateTangents } from "../TangentSpace.js";
import { packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import type { SceneMaterialDesc, SceneMeshDesc } from "../Scene.js";
import type { StaticVertex } from "../SceneData.js";
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

interface AiMesh {
    name: string;
    materialindex: number;
    vertices: number[];
    normals?: number[];
    texturecoords?: number[][];
    faces: number[][];
}

interface AiNode {
    name: string;
    transformation: number[];
    children?: AiNode[];
    meshes?: number[];
}

interface AiScene {
    rootnode: AiNode;
    meshes: AiMesh[];
    materials: AiMaterial[];
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
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; materialNames: string[] }> {
        const ajs = await getAssimp();
        const files = new ajs.FileList();
        files.AddFile("scene.fbx", bytes);
        const result = ajs.ConvertFileList(files, "assjson");
        if (!result.IsSuccess()) throw new RuntimeError(`FbxImporter: assimp failed (${result.GetErrorCode()})`);
        const json = JSON.parse(new TextDecoder().decode(result.GetFile(0).GetContent())) as AiScene;

        // Textures (loaded per unique path; slot decides sRGB like loadMaterialTexture).
        const textureIDs = new Map<string, number>();
        const loadTexture = async (path: string, srgb: boolean): Promise<number | undefined> => {
            const norm = path.replace(/\\/g, "/");
            const key = `${norm}|${srgb}`;
            if (textureIDs.has(key)) return textureIDs.get(key);
            const url = baseUrl ? `${baseUrl}/${norm}` : norm;
            const res = await fetch(url);
            if (!res.ok) return undefined;
            const bitmap = await createImageBitmap(await res.blob(), {
                colorSpaceConversion: "none",
                premultiplyAlpha: "none",
            });
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

        const visit = (node: AiNode, parent: float4x4) => {
            const local = new float4x4();
            for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) local.set(r, c, node.transformation[r * 4 + c]!);
            const world = mulMat(parent, local);
            for (const mi of node.meshes ?? []) {
                const { vertices, indices } = getMesh(mi);
                meshDescs.push({ vertices, indices, materialID: json.meshes[mi]!.materialindex, transform: world });
            }
            for (const child of node.children ?? []) visit(child, world);
        };
        visit(json.rootnode, float4x4.identity());
        if (meshDescs.length === 0) throw new RuntimeError("FbxImporter: no triangle meshes found");

        return { meshes: meshDescs, materials, materialNames };
    }
}
