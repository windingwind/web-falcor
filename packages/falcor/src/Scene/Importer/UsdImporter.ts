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
    opacityThreshold?: number;
    opacityTextureId?: number;
    displacementTextureId?: number;
    useSpecularWorkflow?: boolean;
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
        options: { assumeLinearSpaceTextures?: boolean } = {},
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
                if (m.useSpecularWorkflow) Logger.warning("UsdImporter: Specular workflow is not supported.");
                desc = {
                    name,
                    // PreviewSurfaceConverter forces every material double-sided.
                    header: { materialType: MaterialType.Standard, ior: m.ior ?? 1.5, emissive, doubleSided: true },
                    basic: {
                        baseColor: new float4(dc[0]!, dc[1]!, dc[2]!, 1),
                        specular: new float4(0, m.roughness ?? 0.5, m.metallic ?? 0, 1),
                        emissive: new float3(em[0]!, em[1]!, em[2]!),
                        emissiveFactor: 1,
                    },
                };
                applyUniformOpacity(m, desc);
                const hasTexture = [m.diffuseColorTextureId, m.roughnessTextureId, m.metallicTextureId, m.normalTextureId, m.emissiveColorTextureId, m.opacityTextureId, m.displacementTextureId].some((t) => t !== undefined && t >= 0);
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
                await resolveMaterialTextures(usd, m, desc, textureManager, baseUrl, !!options.assumeLinearSpaceTextures);
            }
        }
        return { meshes, materials, materialNames };
    }
}

/**
 * Mirrors PreviewSurfaceConverter's opacity handling for untextured opacity.
 * Below 1 it is a cutout when an opacity threshold is set (alpha in the base
 * colour, tested against the threshold) and specular transmission of
 * 1 - opacity otherwise. Textured opacity is finished in resolveMaterialTextures.
 */
function applyUniformOpacity(m: UsdMaterial, desc: SceneMaterialDesc): void {
    const opacity = m.opacity ?? 1;
    const threshold = m.opacityThreshold ?? 0;
    const textured = m.opacityTextureId !== undefined && m.opacityTextureId >= 0;
    if (!(opacity < 1 || textured)) return;
    desc.header!.alphaThreshold = threshold;
    if (threshold > 0) {
        const base = desc.basic.baseColor!;
        desc.basic.baseColor = new float4(base.x, base.y, base.z, opacity);
    } else if (!textured) {
        desc.basic.specularTransmission = 1 - opacity;
    }
}

/** Resolves the material's texture slots (mirrors PreviewSurfaceConverter):
 *  baseColor sRGB; roughness+metallic packed into one ORM texture like the
 *  native CreateSpecularTexture kernel (channel r — tinyusdz exposes no
 *  channel selectors); opacity through packBaseColorAlpha or
 *  createSpecularTransmissionTexture; normal, emissive and displacement direct.
 *  The packing kernels run on the CPU here (the port decodes textures there). */
async function resolveMaterialTextures(
    usd: TinyUsdzScene,
    m: UsdMaterial,
    desc: SceneMaterialDesc,
    textureManager: TextureManager,
    baseUrl: string,
    assumeLinear = false,
): Promise<void> {
    const valid = (id: number | undefined): id is number => id !== undefined && id >= 0;
    const load = async (id: number, slotSrgb: boolean): Promise<number | undefined> => {
        try {
            return textureManager.addTexture({ bitmap: await resolveImageBitmap(usd, id, baseUrl), srgb: slotSrgb && !assumeLinear });
        } catch (err) {
            Logger.warning(`UsdImporter: failed to load texture ${id} (${String(err)})`);
            return undefined;
        }
    };
    const threshold = m.opacityThreshold ?? 0;
    const opacityTextured = valid(m.opacityTextureId);
    if (threshold > 0 && (valid(m.diffuseColorTextureId) || opacityTextured) && ((m.opacity ?? 1) < 1 || opacityTextured)) {
        // packBaseColorAlpha: cutout opacity rides in the base colour's alpha.
        try {
            const baseImg = valid(m.diffuseColorTextureId) ? readPixels(await resolveImageBitmap(usd, m.diffuseColorTextureId, baseUrl)) : null;
            const opacityImg = opacityTextured ? readPixels(await resolveImageBitmap(usd, m.opacityTextureId!, baseUrl)) : null;
            const w = baseImg?.width ?? opacityImg!.width;
            const h = baseImg?.height ?? opacityImg!.height;
            const packed = new Uint8ClampedArray(w * h * 4);
            const base = desc.basic.baseColor!;
            // The packed texture is decoded as sRGB, so encode the linear uniform colour.
            const toSrgb = (c: number) => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
            const uniformRgb = [base.x, base.y, base.z].map((c) => Math.round(toSrgb(Math.min(Math.max(c, 0), 1)) * 255));
            const opacityConst = Math.round((m.opacity ?? 1) * 255);
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const i = (y * w + x) * 4;
                    for (let c = 0; c < 3; c++) packed[i + c] = baseImg ? sampleNearest(baseImg, x, y, w, h, c) : uniformRgb[c]!;
                    // tinyusdz exposes no channel selector: the red channel carries opacity.
                    packed[i + 3] = opacityImg ? sampleNearest(opacityImg, x, y, w, h, 0) : opacityConst;
                }
            }
            const bitmap = await createImageBitmap(new ImageData(packed, w, h), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
            desc.basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, textureManager.addTexture({ bitmap, srgb: !assumeLinear }));
        } catch (err) {
            Logger.warning(`UsdImporter: failed to pack base colour and opacity (${String(err)})`);
        }
    } else if (valid(m.diffuseColorTextureId)) {
        const id = await load(m.diffuseColorTextureId, true);
        if (id !== undefined) desc.basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, id);
    }
    if (threshold <= 0 && opacityTextured) {
        // createSpecularTransmissionTexture: textured opacity becomes a grey
        // transmission map of 1 - opacity, with full specular transmission.
        try {
            const opacityImg = readPixels(await resolveImageBitmap(usd, m.opacityTextureId!, baseUrl));
            const out = new Uint8ClampedArray(opacityImg.width * opacityImg.height * 4);
            for (let i = 0; i < out.length; i += 4) {
                const v = 255 - opacityImg.data[i]!;
                out.set([v, v, v, 255], i);
            }
            const bitmap = await createImageBitmap(new ImageData(out, opacityImg.width, opacityImg.height), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
            desc.basic.texTransmission = packTextureHandle(TextureHandleMode.Texture, textureManager.addTexture({ bitmap, srgb: false }));
            desc.basic.specularTransmission = 1;
        } catch (err) {
            Logger.warning(`UsdImporter: failed to build the transmission texture (${String(err)})`);
        }
    }
    if (valid(m.displacementTextureId)) {
        const id = await load(m.displacementTextureId, false);
        if (id !== undefined) desc.basic.texDisplacement = packTextureHandle(TextureHandleMode.Texture, id);
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

/** Nearest texel of `img` for pixel (x, y) of a w x h target, channel c. */
function sampleNearest(img: { data: Uint8ClampedArray; width: number; height: number }, x: number, y: number, w: number, h: number, c: number): number {
    const sx = Math.min(Math.floor((x * img.width) / w), img.width - 1);
    const sy = Math.min(Math.floor((y * img.height) / h), img.height - 1);
    return img.data[(sy * img.width + sx) * 4 + c]!;
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
