/**
 * USD importer (subset of plugins/importers/USDImporter) via tinyusdz-wasm:
 * UsdGeomMesh + xform hierarchy + UsdPreviewSurface materials incl.
 * UsdUVTexture baseColor (sRGB, V-flipped st). Lights, cameras and UsdSkel
 * are not exposed by the tinyusdz RenderScene API yet (docs §8.4). Assets
 * load as .usda/.usdc/.usdz.
 */

import type { SceneMeshDesc, SceneMaterialDesc } from "../Scene.js";
import type { TextureManager } from "../Material/TextureManager.js";
import { MaterialType, packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import { generateTangents } from "../TangentSpace.js";
import { type StaticVertex } from "../SceneData.js";
import { float2, float3, float4 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";

interface UsdNode {
    primName: string;
    nodeType: string;
    contentId: number;
    localMatrix?: ArrayLike<number>;
    children: UsdNode[];
}

interface UsdMesh {
    points: Float32Array;
    faceVertexIndices: Uint32Array;
    normals?: Float32Array;
    texcoords?: Float32Array;
    materialId?: number;
}

interface UsdMaterial {
    name?: string;
    diffuseColorTextureId?: number;
    roughnessTextureId?: number;
    metallicTextureId?: number;
    normalTextureId?: number;
    emissiveColorTextureId?: number;
    diffuseColor?: ArrayLike<number>;
    roughness?: number;
    metallic?: number;
    ior?: number;
    emissiveColor?: ArrayLike<number>;
    opacity?: number;
}

interface UsdImage {
    uri?: string;
    bufferId: number;
    data?: Uint8Array;
    decoded?: boolean;
    width?: number;
    height?: number;
    channels?: number;
}

interface TinyUsdzScene {
    loadFromBinary(bytes: Uint8Array, path: string): boolean;
    error(): string;
    getDefaultRootNode(): UsdNode;
    getMesh(contentId: number): UsdMesh;
    getMaterial(materialId: number): UsdMaterial;
    getTexture(textureId: number): { textureImageId: number };
    getImage(imageId: number): UsdImage;
}

interface TinyUsdzModule {
    TinyUSDZLoaderNative: new () => TinyUsdzScene;
}

let modulePromise: Promise<TinyUsdzModule> | null = null;

/** Loads the tinyusdz wasm module (served from node_modules, like pyodide). */
function loadTinyUsdz(baseUrl = "/node_modules/tinyusdz"): Promise<TinyUsdzModule> {
    modulePromise ??= import(/* @vite-ignore */ `${baseUrl}/tinyusdz.js`).then(
        (m: { default: () => Promise<TinyUsdzModule> }) => m.default(),
    );
    return modulePromise;
}

/** USD matrices are row-major with row-vector convention (p' = p*M);
 *  the web float4x4 applies column vectors (p' = M*p) -> transpose. */
function usdToWebMatrix(a: ArrayLike<number>): float4x4 {
    const m = new float4x4();
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) m.set(r, c, a[c * 4 + r]!);
    return m;
}

/** Area-weighted smooth normals for meshes without authored normals. */
function computeSmoothNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
    const n = new Float32Array(positions.length);
    for (let t = 0; t < indices.length; t += 3) {
        const [i0, i1, i2] = [indices[t]!, indices[t + 1]!, indices[t + 2]!];
        const ax = positions[i1 * 3]! - positions[i0 * 3]!;
        const ay = positions[i1 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const az = positions[i1 * 3 + 2]! - positions[i0 * 3 + 2]!;
        const bx = positions[i2 * 3]! - positions[i0 * 3]!;
        const by = positions[i2 * 3 + 1]! - positions[i0 * 3 + 1]!;
        const bz = positions[i2 * 3 + 2]! - positions[i0 * 3 + 2]!;
        const cx = ay * bz - az * by;
        const cy = az * bx - ax * bz;
        const cz = ax * by - ay * bx;
        for (const i of [i0, i1, i2]) {
            n[i * 3] = n[i * 3]! + cx;
            n[i * 3 + 1] = n[i * 3 + 1]! + cy;
            n[i * 3 + 2] = n[i * 3 + 2]! + cz;
        }
    }
    for (let i = 0; i < n.length; i += 3) {
        const len = Math.hypot(n[i]!, n[i + 1]!, n[i + 2]!) || 1;
        n[i]! /= len;
        n[i + 1]! /= len;
        n[i + 2]! /= len;
    }
    return n;
}

export class UsdImporter {
    /** Parses USD (usda/usdc/usdz) into scene descriptors (device-free). */
    static async parseToDescs(
        bytes: Uint8Array,
        textureManager?: TextureManager,
        baseUrl = "",
        excludePrims?: Set<string>,
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; materialNames: string[] }> {
        const native = await loadTinyUsdz();
        const usd = new native.TinyUSDZLoaderNative();
        if (!usd.loadFromBinary(bytes, "scene.usd")) {
            throw new RuntimeError(`UsdImporter: failed to parse USD (${usd.error()})`);
        }

        const meshes: SceneMeshDesc[] = [];
        const materials: SceneMaterialDesc[] = [];
        const materialNames: string[] = [];
        const materialIndex = new Map<number, number>();
        const textureJobs: { desc: SceneMaterialDesc; material: UsdMaterial }[] = [];

        const getOrAddMaterial = (materialId: number | undefined): number => {
            const id = materialId ?? -1;
            const existing = materialIndex.get(id);
            if (existing !== undefined) return existing;
            let desc: SceneMaterialDesc;
            let name = "";
            if (id >= 0) {
                const m = usd.getMaterial(id);
                name = m.name ?? "";
                const dc = m.diffuseColor ?? [0.18, 0.18, 0.18];
                const em = m.emissiveColor ?? [0, 0, 0];
                const emissive = em[0]! !== 0 || em[1]! !== 0 || em[2]! !== 0;
                desc = {
                    name,
                    header: { materialType: MaterialType.Standard, ior: m.ior ?? 1.5, emissive },
                    basic: {
                        baseColor: new float4(dc[0]!, dc[1]!, dc[2]!, m.opacity ?? 1),
                        specular: new float4(0, m.roughness ?? 0.5, m.metallic ?? 0, 0),
                        emissive: new float3(em[0]!, em[1]!, em[2]!),
                        emissiveFactor: 1,
                    },
                };
                const hasTexture = [m.diffuseColorTextureId, m.roughnessTextureId, m.metallicTextureId, m.normalTextureId, m.emissiveColorTextureId].some((t) => t !== undefined && t >= 0);
                if (hasTexture) textureJobs.push({ desc, material: m });
            } else {
                // UsdPreviewSurface fallback (18% gray).
                desc = {
                    header: { materialType: MaterialType.Standard },
                    basic: { baseColor: new float4(0.18, 0.18, 0.18, 1), specular: new float4(0, 0.5, 0, 0) },
                };
            }
            const index = materials.length;
            materials.push(desc);
            materialNames.push(name);
            materialIndex.set(id, index);
            return index;
        };

        const walk = (node: UsdNode, parentWorld: float4x4): void => {
            let world = parentWorld;
            if (node.localMatrix && node.localMatrix.length === 16) {
                world = mulMat(parentWorld, usdToWebMatrix(node.localMatrix));
            }
            if (node.nodeType === "mesh" && !excludePrims?.has(node.primName)) {
                const mesh = usd.getMesh(node.contentId);
                const positions = mesh.points;
                const indices = new Uint32Array(mesh.faceVertexIndices);
                const normals = mesh.normals && mesh.normals.length === positions.length ? mesh.normals : computeSmoothNormals(positions, indices);
                const uvs = mesh.texcoords;
                const vertexCount = positions.length / 3;
                const vertices: StaticVertex[] = new Array(vertexCount);
                for (let i = 0; i < vertexCount; i++) {
                    vertices[i] = {
                        position: new float3(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!),
                        normal: new float3(normals[i * 3]!, normals[i * 3 + 1]!, normals[i * 3 + 2]!),
                        tangent: new float4(0, 0, 0, 0),
                        // USD st has a bottom-left origin; Falcor samples top-down
                        // images with raw st (native parity) -> flip V.
                        texCrd: uvs && uvs.length === vertexCount * 2 ? new float2(uvs[i * 2]!, 1 - uvs[i * 2 + 1]!) : new float2(0, 0),
                    };
                }
                generateTangents(vertices, indices);
                meshes.push({ vertices, indices, materialID: getOrAddMaterial(mesh.materialId), transform: world.clone() });
            } else if (node.nodeType !== "xform" && node.nodeType !== "") {
                Logger.warning(`UsdImporter: prim type '${node.nodeType}' ('${node.primName}') not supported (skipped)`);
            }
            for (const child of node.children ?? []) walk(child, world);
        };

        walk(usd.getDefaultRootNode(), float4x4.identity());

        // Resolve UsdUVTexture images (URI, embedded-encoded, or pre-decoded).
        if (textureManager) {
            for (const { desc, material: m } of textureJobs) {
                await resolveMaterialTextures(usd, m, desc, textureManager, baseUrl);
            }
        }
        return { meshes, materials, materialNames };
    }
}

/** Resolves the material's texture slots (mirrors PreviewSurfaceConverter):
 *  baseColor sRGB; roughness+metallic packed into one ORM texture like the
 *  native CreateSpecularTexture kernel (channel r — tinyusdz exposes no
 *  channel selectors); normal + emissive direct. */
async function resolveMaterialTextures(
    usd: TinyUsdzScene,
    m: UsdMaterial,
    desc: SceneMaterialDesc,
    textureManager: TextureManager,
    baseUrl: string,
): Promise<void> {
    const valid = (id: number | undefined): id is number => id !== undefined && id >= 0;
    const load = async (id: number, srgb: boolean): Promise<number | undefined> => {
        try {
            return textureManager.addTexture({ bitmap: await resolveImageBitmap(usd, id, baseUrl), srgb });
        } catch (err) {
            Logger.warning(`UsdImporter: failed to load texture ${id} (${String(err)})`);
            return undefined;
        }
    };
    if (valid(m.diffuseColorTextureId)) {
        const id = await load(m.diffuseColorTextureId, true);
        if (id !== undefined) desc.basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, id);
    }
    if (valid(m.roughnessTextureId) || valid(m.metallicTextureId)) {
        try {
            const rough = valid(m.roughnessTextureId) ? readPixels(await resolveImageBitmap(usd, m.roughnessTextureId, baseUrl)) : null;
            const metal = valid(m.metallicTextureId) ? readPixels(await resolveImageBitmap(usd, m.metallicTextureId, baseUrl)) : null;
            const w = Math.max(rough?.width ?? 0, metal?.width ?? 0);
            const h = Math.max(rough?.height ?? 0, metal?.height ?? 0);
            const orm = new Uint8ClampedArray(w * h * 4);
            const roughConst = Math.round((desc.basic.specular?.y ?? 0.5) * 255);
            const metalConst = Math.round((desc.basic.specular?.z ?? 0) * 255);
            const sample = (img: { data: Uint8ClampedArray; width: number; height: number } | null, x: number, y: number, fallback: number): number => {
                if (!img) return fallback;
                const sx = Math.min(Math.floor((x * img.width) / w), img.width - 1);
                const sy = Math.min(Math.floor((y * img.height) / h), img.height - 1);
                return img.data[(sy * img.width + sx) * 4]!;
            };
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const i = (y * w + x) * 4;
                    orm[i + 1] = sample(rough, x, y, roughConst);
                    orm[i + 2] = sample(metal, x, y, metalConst);
                    orm[i + 3] = 255;
                }
            }
            const bitmap = await createImageBitmap(new ImageData(orm, w, h));
            desc.basic.texSpecular = packTextureHandle(TextureHandleMode.Texture, textureManager.addTexture({ bitmap, srgb: false }));
        } catch (err) {
            Logger.warning(`UsdImporter: failed to pack spec texture (${String(err)})`);
        }
    }
    if (valid(m.normalTextureId)) {
        const id = await load(m.normalTextureId, false);
        if (id !== undefined) desc.basic.texNormalMap = packTextureHandle(TextureHandleMode.Texture, id);
    }
    if (valid(m.emissiveColorTextureId)) {
        const id = await load(m.emissiveColorTextureId, true);
        if (id !== undefined) {
            desc.basic.texEmissive = packTextureHandle(TextureHandleMode.Texture, id);
            if (desc.header) desc.header.emissive = true;
        }
    }
}

/** Reads an ImageBitmap back to pixels (packing input for the ORM texture). */
function readPixels(bitmap: ImageBitmap): { data: Uint8ClampedArray; width: number; height: number } {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
    ctx.drawImage(bitmap, 0, 0);
    return { data: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, height: bitmap.height };
}

/** Resolves a UsdUVTexture to an ImageBitmap (mirrors the three tinyusdz
 *  image states: URI-only, embedded-encoded, embedded-decoded). */
async function resolveImageBitmap(usd: TinyUsdzScene, textureId: number, baseUrl: string): Promise<ImageBitmap> {
    const image = usd.getImage(usd.getTexture(textureId).textureImageId);
    const opts: ImageBitmapOptions = { colorSpaceConversion: "none", premultiplyAlpha: "none" };
    if (image.uri && image.bufferId === -1) {
        const url = baseUrl ? `${baseUrl}/${image.uri}` : image.uri;
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`fetch '${url}' (${res.status})`);
        return createImageBitmap(await res.blob(), opts);
    }
    if (image.bufferId >= 0 && image.data) {
        if (image.decoded && image.width && image.height) {
            // Raw pixels; expand to RGBA for ImageData.
            const channels = image.channels ?? 4;
            const count = image.width * image.height;
            const rgba = new Uint8ClampedArray(count * 4);
            for (let i = 0; i < count; i++) {
                rgba[i * 4] = image.data[i * channels]!;
                rgba[i * 4 + 1] = image.data[i * channels + (channels > 1 ? 1 : 0)]!;
                rgba[i * 4 + 2] = image.data[i * channels + (channels > 2 ? 2 : 0)]!;
                rgba[i * 4 + 3] = channels > 3 ? image.data[i * channels + 3]! : 255;
            }
            return createImageBitmap(new ImageData(rgba, image.width, image.height));
        }
        return createImageBitmap(new Blob([image.data.slice().buffer as ArrayBuffer]), opts);
    }
    throw new RuntimeError("unresolvable image (no uri, no buffer)");
}
