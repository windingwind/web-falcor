/**
 * USD importer (subset of plugins/importers/USDImporter) via tinyusdz-wasm:
 * UsdGeomMesh + xform hierarchy + UsdPreviewSurface materials incl.
 * UsdUVTexture baseColor (sRGB, V-flipped st). Lights, cameras and UsdSkel
 * are not exposed by the tinyusdz RenderScene API yet (docs §8.4). Assets
 * load as .usda/.usdc/.usdz.
 */

import type { SceneMeshDesc, SceneMaterialDesc, SceneMetadata } from "../Scene.js";
import type { TextureManager } from "../Material/TextureManager.js";
import { MaterialType, packTextureHandle, TextureHandleMode } from "../Material/MaterialData.js";
import { generateTangents } from "../TangentSpace.js";
import { type StaticVertex } from "../SceneData.js";
import { float2, float3, float4 } from "../../Utils/Math/Vector.js";
import { float4x4, inverse, mulMat, transformPoint } from "../../Utils/Math/Matrix.js";
import { RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";
import { decomposeTRS, type AnimationChannel, type SceneNode, type SkinDesc } from "../Animation/SceneAnimation.js";
import { loadOpenSubdiv, tessellateUsdMesh, type TessellatedMesh } from "./Subdivision.js";
import { refinedCorners, triangulateUsdMesh, type CornerMesh } from "./UsdTriangulate.js";
import { extractUsdCamerasAndLights, extractUsdDisplayColors, extractUsdMaterialBindings, extractUsdMaterialTextures, extractUsdPointInstancers, extractUsdMeshes, extractUsdSkeletons, extractUsdXformAnimations, sampleAt, usdRenderSettings, usdTimeCodesPerSecond, usdaStageInfo, usdChannelIndex, usdStageRootTransform, usdTexCoordTransform, type UsdaCamera, type UsdaDomeLight, type UsdaSkeleton, type UsdaSubdivMesh, type UsdaTextureInput, type UsdaXformAnimation } from "./UsdaScene.js";
import type { AnalyticLight } from "../SceneData.js";

interface UsdNode {
    primName: string;
    absPath?: string;
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

/** The USDA text of a USD file: usda as is, usdc/usdz through tinyusdz's layer printer (null if unavailable). */
function usdLayerText(native: TinyUsdzModule, bytes: Uint8Array): string | null {
    const head = new TextDecoder().decode(bytes.subarray(0, 5));
    if (head === "#usda") return new TextDecoder().decode(bytes);
    try {
        const layer = new native.TinyUSDZLoaderNative() as unknown as { loadAsLayerFromBinary?(b: Uint8Array, p: string): boolean; layerToString?(): string };
        if (layer.loadAsLayerFromBinary?.(bytes, "layer.usd") && layer.layerToString) return layer.layerToString();
    } catch {
        /* fall through */
    }
    Logger.warning("UsdImporter: this USD file's layer text is unavailable; cameras, lights and stage units are not imported.");
    return null;
}

/**
 * Splits a triangulated mesh per corner with flat normals: per coarse polygon (the normalized
 * sum of its fan's cross products, as native's triangulate()) when the authored topology matches
 * tinyusdz's fan triangulation, else per triangle.
 */
function flatNormals(points: Float32Array, indices: Uint32Array, uvs: Float32Array | undefined, usda: UsdaSubdivMesh | undefined): { positions: Float32Array; indices: Uint32Array; normals: Float32Array; uvs: Float32Array | undefined } {
    const triCount = indices.length / 3;
    const faceOfTri = new Int32Array(triCount).map((_, t) => t);
    let faceCount = triCount;
    if (usda && usda.faceVertexCounts.reduce((n, c) => n + Math.max(0, c - 2), 0) === triCount) {
        let t = 0;
        usda.faceVertexCounts.forEach((c, f) => {
            for (let k = 0; k < c - 2; k++) faceOfTri[t++] = f;
        });
        faceCount = usda.faceVertexCounts.length;
    }
    const faceNormal = new Float32Array(faceCount * 3);
    for (let t = 0; t < triCount; t++) {
        const [a, b, c] = [indices[t * 3]!, indices[t * 3 + 1]!, indices[t * 3 + 2]!].map((i) => [points[i * 3]!, points[i * 3 + 1]!, points[i * 3 + 2]!]);
        const e1 = [b![0]! - a![0]!, b![1]! - a![1]!, b![2]! - a![2]!];
        const e2 = [c![0]! - a![0]!, c![1]! - a![1]!, c![2]! - a![2]!];
        const f = faceOfTri[t]!;
        faceNormal[f * 3]! += e1[1]! * e2[2]! - e1[2]! * e2[1]!;
        faceNormal[f * 3 + 1]! += e1[2]! * e2[0]! - e1[0]! * e2[2]!;
        faceNormal[f * 3 + 2]! += e1[0]! * e2[1]! - e1[1]! * e2[0]!;
    }
    const positions = new Float32Array(indices.length * 3);
    const normals = new Float32Array(indices.length * 3);
    const outUvs = uvs && uvs.length === points.length / 3 * 2 ? new Float32Array(indices.length * 2) : undefined;
    for (let c = 0; c < indices.length; c++) {
        const i = indices[c]!;
        const f = faceOfTri[Math.floor(c / 3)]!;
        const len = Math.hypot(faceNormal[f * 3]!, faceNormal[f * 3 + 1]!, faceNormal[f * 3 + 2]!) || 1;
        for (let k = 0; k < 3; k++) {
            positions[c * 3 + k] = points[i * 3 + k]!;
            normals[c * 3 + k] = faceNormal[f * 3 + k]! / len;
        }
        if (outUvs) outUvs.set([uvs![i * 2]!, uvs![i * 2 + 1]!], c * 2);
    }
    return { positions, indices: Uint32Array.from(indices.keys()), normals, uvs: outUvs };
}

/** Static vertices from per-corner attributes (normals normalized, as native's keyframes). */
function cornerVertices(m: CornerMesh, toTexCrd: (s: number, t: number) => [number, number]): StaticVertex[] {
    const unit = (c: number) => {
        const [x, y, z] = [m.normals[c * 3]!, m.normals[c * 3 + 1]!, m.normals[c * 3 + 2]!];
        const l = Math.hypot(x, y, z) || 1;
        return new float3(x / l, y / l, z / l);
    };
    return Array.from({ length: m.positions.length / 3 }, (_, c) => ({
        position: new float3(m.positions[c * 3]!, m.positions[c * 3 + 1]!, m.positions[c * 3 + 2]!),
        normal: unit(c),
        tangent: new float4(0, 0, 0, 0),
        texCrd: m.uvs ? new float2(...toTexCrd(m.uvs[c * 2]!, m.uvs[c * 2 + 1]!)) : new float2(0, 0),
    }));
}

/** Static vertices of a refined mesh; face-varying and uniform texcoords split vertices per triangle corner. */
function refinedVertices(m: TessellatedMesh, toTexCrd: (s: number, t: number) => [number, number]): { vertices: StaticVertex[]; indices: Uint32Array } {
    const vertex = (p: number, uv: number | null): StaticVertex => ({
        position: new float3(m.positions[p * 3]!, m.positions[p * 3 + 1]!, m.positions[p * 3 + 2]!),
        normal: new float3(m.normals[p * 3]!, m.normals[p * 3 + 1]!, m.normals[p * 3 + 2]!),
        tangent: new float4(0, 0, 0, 0),
        texCrd: uv !== null && m.uvs ? new float2(...toTexCrd(m.uvs[uv * 2]!, m.uvs[uv * 2 + 1]!)) : new float2(0, 0),
    });
    if (m.uvInterp !== "faceVarying" && m.uvInterp !== "uniform") {
        const vertices = Array.from({ length: m.positions.length / 3 }, (_, p) => vertex(p, m.uvs ? p : null));
        return { vertices, indices: m.indices };
    }
    // Uniform texcoords: one per coarse face, in the order the faces were tessellated.
    const faceUv = new Map<number, number>();
    for (const f of m.coarseFaces) if (!faceUv.has(f)) faceUv.set(f, faceUv.size);
    const vertices = Array.from(m.indices, (p, c) => vertex(p, m.uvInterp === "faceVarying" ? c : faceUv.get(m.coarseFaces[Math.floor(c / 3)]!)!));
    return { vertices, indices: Uint32Array.from(vertices.keys()) };
}

/** tinyusdz's layer API (composition), on the same loader class. */
interface TinyUsdzLayer {
    loadAsLayerFromBinary(bytes: Uint8Array, path: string): boolean;
    layerToString(): string;
    layerToRenderScene(): boolean;
    extractSublayerAssetPaths(): unknown;
    extractReferencesAssetPaths(): unknown;
    extractPayloadAssetPaths(): unknown;
    composeSublayers(): boolean;
    hasReferences(): boolean;
    composeReferences(): boolean;
    hasPayload(): boolean;
    composePayload(): boolean;
    hasInherits(): boolean;
    composeInherits(): boolean;
    hasVariants(): boolean;
    composeVariants(): boolean;
    setAsset(path: string, bytes: Uint8Array): void;
    error(): string;
}

/** Embind vectors or arrays as arrays. */
const toArray = (v: unknown): string[] => {
    if (Array.isArray(v)) return v as string[];
    const vec = v as { size?(): number; get(i: number): string } | null;
    return vec?.size ? Array.from({ length: vec.size() }, (_, i) => vec.get(i)) : [];
};

/**
 * Composes a USD layer as a stage does (sublayers, then inherits, variants, references and
 * payloads until none remain), fetching external layers relative to `baseUrl`. Returns the
 * composed layer's text (the caller builds the render scene), or null if the file has no
 * composition arcs.
 */
async function composeUsdLayer(usd: TinyUsdzLayer, bytes: Uint8Array, baseUrl: string): Promise<string | null> {
    if (!usd.loadAsLayerFromBinary(bytes, "scene.usd")) return null;
    const sublayers = toArray(usd.extractSublayerAssetPaths());
    // tinyusdz's hasVariants() reports none, so variant sets are found in the text.
    const hasVariantSets = () => /\bvariantSet\s+"/.test(usd.layerToString());
    if (sublayers.length === 0 && !usd.hasReferences() && !usd.hasPayload() && !usd.hasInherits() && !hasVariantSets()) return null;
    const fetched = new Set<string>();
    const fetchAssets = async (paths: string[]) => {
        await Promise.all(
            // Internal references have no asset path.
            paths.filter((p) => p && !fetched.has(p)).map(async (p) => {
                fetched.add(p);
                const url = /^([a-z]+:|\/)/i.test(p) || !baseUrl ? p : `${baseUrl}/${p.replace(/^\.\//, "")}`;
                const res = await fetch(url);
                if (!res.ok) throw new RuntimeError(`UsdImporter: can't find layer '${p}' (tried '${url}', ${res.status})`);
                usd.setAsset(p, new Uint8Array(await res.arrayBuffer()));
            }),
        );
    };
    await fetchAssets(sublayers);
    if (!usd.composeSublayers()) throw new RuntimeError(`UsdImporter: failed to compose sublayers (${usd.error()})`);
    for (let i = 0; i < 16; i++) {
        const [refs, payload, inherits, variants] = [usd.hasReferences(), usd.hasPayload(), usd.hasInherits(), hasVariantSets()];
        if (!refs && !payload && !inherits && !variants) break;
        if (inherits && !usd.composeInherits()) throw new RuntimeError(`UsdImporter: failed to compose inherits (${usd.error()})`);
        if (variants && !usd.composeVariants()) throw new RuntimeError(`UsdImporter: failed to compose variants (${usd.error()})`);
        if (refs) {
            await fetchAssets(toArray(usd.extractReferencesAssetPaths()));
            if (!usd.composeReferences()) throw new RuntimeError(`UsdImporter: failed to compose references (${usd.error()})`);
        }
        if (payload) {
            await fetchAssets(toArray(usd.extractPayloadAssetPaths()));
            if (!usd.composePayload()) throw new RuntimeError(`UsdImporter: failed to compose payloads (${usd.error()})`);
        }
    }
    return usd.layerToString();
}

/** A mesh's first display color (on the prim, else its parent), as native's default material uses; [0.7, 0.7, 0.7] if none. */
function usdDisplayColor(colors: Map<string, [number, number, number]>, meshPath: string | undefined): [number, number, number] {
    const parent = meshPath?.slice(0, meshPath.lastIndexOf("/"));
    return (meshPath && colors.get(meshPath)) || (parent && colors.get(parent)) || [0.7, 0.7, 0.7];
}

/** The stage's bounding box center and diagonal in meters (USD space: no up-axis rotation). */
export interface UsdStageBounds {
    center: float3;
    diagonal: number;
}

export class UsdImporter {
    /** Parses USD (usda/usdc/usdz) into scene descriptors (device-free). */
    static async parseToDescs(
        bytes: Uint8Array,
        textureManager?: TextureManager,
        baseUrl = "",
        excludePrims?: Set<string>,
        options: {
            assumeLinearSpaceTextures?: boolean;
            /** Per-prim Settings attributes (native's "refinementLevel" override by mesh path). */
            settings?: { getAttribute(path: string, name: string, fallback: number): unknown };
        } = {},
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; materialNames: string[]; cameras: UsdaCamera[]; lights: AnalyticLight[]; domeLight: UsdaDomeLight | null; stage: UsdStageBounds | null; metadata: SceneMetadata | null; nodes: SceneNode[]; animations: AnimationChannel[] }> {
        const native = await loadTinyUsdz();
        let usd = new native.TinyUSDZLoaderNative();
        // Files with composition arcs are composed first.
        const composedText = await composeUsdLayer(usd as unknown as TinyUsdzLayer, bytes, baseUrl);
        // Cameras, lights and stage metadata come from the layer's text (RenderScene has none of them).
        const layerText = composedText ?? usdLayerText(native, bytes);
        // tinyusdz's RenderScene conversion crashes on skel joint primvars; they are read from the
        // layer text instead, so its scene is built from the text without them.
        const skelPrimvars = /^\s*[\w\[\]]+\s+primvars:skel:\w+\s*=\s*(\[[^\]]*\]|\([^)]*\))(\s*\([^)]*\))?/gm;
        if (layerText && skelPrimvars.test(layerText)) {
            usd = new native.TinyUSDZLoaderNative();
            if (!usd.loadFromBinary(new TextEncoder().encode(layerText.replace(skelPrimvars, "")), "scene.usda")) throw new RuntimeError(`UsdImporter: failed to parse USD (${usd.error()})`);
        } else if (composedText !== null) {
            const layer = usd as unknown as TinyUsdzLayer;
            if (!layer.layerToRenderScene()) throw new RuntimeError(`UsdImporter: failed to build the composed scene (${layer.error()})`);
        } else {
            usd = new native.TinyUSDZLoaderNative();
            if (!usd.loadFromBinary(bytes, "scene.usd")) throw new RuntimeError(`UsdImporter: failed to parse USD (${usd.error()})`);
        }
        const stageInfo = layerText ? usdaStageInfo(layerText) : null;
        const rootXform = stageInfo ? usdStageRootTransform(stageInfo) : float4x4.identity();
        const extracted = layerText ? extractUsdCamerasAndLights(layerText) : { cameras: [], lights: [], domeLight: null };
        // Channel selectors, color spaces and st transforms of textured inputs (tinyusdz drops them).
        const materialTextures = layerText ? extractUsdMaterialTextures(layerText) : new Map<string, Map<string, UsdaTextureInput>>();
        // tinyusdz materials carry no name: find them through the meshes' bindings.
        const bindings = layerText ? extractUsdMaterialBindings(layerText) : new Map<string, string>();
        const displayColors = layerText ? extractUsdDisplayColors(layerText) : new Map<string, [number, number, number]>();

        const meshes: SceneMeshDesc[] = [];
        const materials: SceneMaterialDesc[] = [];
        const materialNames: string[] = [];
        const materialIndex = new Map<number | string, number>();
        const texCoordTransforms: ((s: number, t: number) => [number, number])[] = [];
        const textureJobs: { desc: SceneMaterialDesc; material: UsdMaterial; inputs: Map<string, UsdaTextureInput> }[] = [];

        const getOrAddMaterial = (materialId: number | undefined, materialPath: string | undefined, meshPath?: string): number => {
            const id = materialId ?? -1;
            // Unbound meshes share native's default material per display color.
            const color = id < 0 ? usdDisplayColor(displayColors, meshPath) : undefined;
            const key = color ? `default:${color.join(",")}` : id;
            const existing = materialIndex.get(key);
            if (existing !== undefined) return existing;
            let desc: SceneMaterialDesc;
            let name = "";
            let texTransform: UsdaTextureInput["transform"];
            if (id >= 0) {
                const m = usd.getMaterial(id);
                name = m.name || (materialPath?.slice(materialPath.lastIndexOf("/") + 1) ?? "");
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
                const inputs = (materialPath && materialTextures.get(materialPath)) || new Map<string, UsdaTextureInput>();
                const emissiveScale = inputs.get("emissivecolor")?.scale;
                if (emissiveScale && emissiveScale.some((v) => v !== 1)) {
                    if (emissiveScale.some((v) => v !== emissiveScale[0])) Logger.warning(`UsdPreviewSurface '${name}' input 'emissiveColor' specifies a vector texture value scale. Applying red component to all channels.`);
                    desc.basic.emissiveFactor = emissiveScale[0];
                }
                // ConvertedTexTransform: the first non-identity st transform applies to every texture.
                const isIdentity = (t: UsdaTextureInput["transform"]) => !t || (t.scale[0] === 1 && t.scale[1] === 1 && t.rotation === 0 && t.translation[0] === 0 && t.translation[1] === 0);
                for (const input of inputs.values()) {
                    if (isIdentity(input.transform)) continue;
                    if (!texTransform) texTransform = input.transform;
                    else if (JSON.stringify(texTransform) !== JSON.stringify(input.transform)) Logger.warning(`Shader input '${input.texture}' specifies a texture transform that differs from that used on another texture, which is not supported. Applying the first encountered non-idenity transform to all textures.`);
                }
                const hasTexture = [m.diffuseColorTextureId, m.roughnessTextureId, m.metallicTextureId, m.normalTextureId, m.emissiveColorTextureId, m.opacityTextureId, m.displacementTextureId].some((t) => t !== undefined && t >= 0);
                if (hasTexture) textureJobs.push({ desc, material: m, inputs });
            } else {
                // ImporterContext::getDefaultMaterial: the display color, roughness 0.3, double-sided.
                const c = color ?? [0.7, 0.7, 0.7];
                name = `default-mesh-${[...materialIndex.keys()].filter((k) => typeof k === "string").length}`;
                desc = {
                    name,
                    header: { materialType: MaterialType.Standard, ior: 1.5, doubleSided: true },
                    basic: { baseColor: new float4(c[0]!, c[1]!, c[2]!, 1), specular: new float4(0, 0.3, 0, 1) },
                };
            }
            const index = materials.length;
            materials.push(desc);
            materialNames.push(name);
            materialIndex.set(key, index);
            // Native pre-transforms texcoords by the material's texture transform: a V flip (and
            // any UsdTransform2d) for UsdPreviewSurface materials, none for the default material.
            texCoordTransforms.push(id >= 0 ? usdTexCoordTransform(texTransform) : (s, t) => [s, t]);
            return index;
        };

        // PointInstancers (tinyusdz keeps their prototypes as plain children): the prototypes
        // render once per instance and nowhere else, as natively.
        const instancers = new Map((layerText ? extractUsdPointInstancers(layerText) : []).map((i) => [i.path, i]));
        const nodesByPath = new Map<string, UsdNode>();
        const indexNodes = (n: UsdNode) => {
            if (n.absPath) nodesByPath.set(n.absPath, n);
            (n.children ?? []).forEach(indexNodes);
        };
        indexNodes(usd.getDefaultRootNode());
        // Subdivision surfaces (convertMeshGeomData's refinement level, then tessellate()).
        const usdaMeshes = layerText ? extractUsdMeshes(layerText) : new Map<string, UsdaSubdivMesh>();
        const stageRefinement = layerText ? (usdRenderSettings(layerText)?.refinementLevel ?? 0) : 0;
        const refinementLevels = new Map<string, number>();
        for (const [path, m] of usdaMeshes) {
            if (m.scheme === "none") continue;
            let level = Number(options.settings?.getAttribute(path, "refinementLevel", m.refinementLevel ?? stageRefinement) ?? m.refinementLevel ?? stageRefinement);
            if (level > 0 && m.skinned) {
                Logger.warning(`Skipping subdividing skinned mesh '${path}'.`);
                level = 0;
            }
            if (level > 0) refinementLevels.set(path, Number(level));
        }
        const osd = refinementLevels.size > 0 ? await loadOpenSubdiv() : null;
        // UsdSkel skeletons: joints become bone nodes with native's per-bone animations. Native's
        // skinning pass cancels the skeleton's world transform and renders the result with the
        // mesh's, so skinned = meshWorld * joint * inverse bind: root bones hang off a node with the
        // mesh's world (one bone set per skeleton and mesh world).
        const { skeletons, bindings: skelBindings } = layerText ? extractUsdSkeletons(layerText) : { skeletons: new Map<string, UsdaSkeleton>(), bindings: new Map<string, string>() };
        const tcps = layerText ? usdTimeCodesPerSecond(layerText) : 24;
        const boneNodes = new Map<string, number[]>();
        const usdSkin = (m: UsdaSubdivMesh, skel: UsdaSkeleton, pointOfCorner: Uint32Array, meshWorld: float4x4, path: string): SkinDesc => {
            const key = `${skel.path} ${meshWorld.toArray().join(",")}`;
            let ids = boneNodes.get(key);
            if (!ids) {
                ids = [];
                const skelRoot = nodes.length;
                nodes.push({ parent: -1, ...decomposeTRS(meshWorld) });
                for (let i = 0; i < skel.joints.length; i++) {
                    const parentPath = skel.joints[i]!.slice(0, Math.max(0, skel.joints[i]!.lastIndexOf("/")));
                    const parent = skel.joints.indexOf(parentPath);
                    const id = nodes.length;
                    nodes.push({ parent: parent >= 0 ? ids[parent]! : skelRoot, ...decomposeTRS(skel.rest[i]!) });
                    ids.push(id);
                    const clip = new Set(animations.map((c) => c.clip)).size;
                    const times = Float32Array.from(skel.anim!.times.map((t) => t / tcps));
                    const channel = (p: AnimationChannel["path"], values: number[]) => animations.push({ nodeID: id, path: p, times, values: Float32Array.from(values), interp: "LINEAR", clip });
                    channel("translation", skel.anim!.translation.flatMap((pose) => [pose[i]!.x, pose[i]!.y, pose[i]!.z]));
                    channel("rotation", skel.anim!.rotation.flatMap((pose) => [pose[i]!.x, pose[i]!.y, pose[i]!.z, pose[i]!.w]));
                    channel("scale", skel.anim!.scale.flatMap((pose) => [pose[i]!.x, pose[i]!.y, pose[i]!.z]));
                }
                boneNodes.set(key, ids);
            }
            const { indices, weights, elementSize, joints } = m.skin!;
            if (elementSize > 4) Logger.warning(`Mesh '${path}' contains more than 4 bones per vertex (${elementSize}). Ignoring extra data.`);
            // A mesh's skel:joints may name a subset of the skeleton's joints.
            const remap = joints ? joints.map((j) => skel.joints.indexOf(j)) : null;
            const boneIDs = new Uint32Array(pointOfCorner.length * 4);
            const boneWeights = new Float32Array(pointOfCorner.length * 4);
            pointOfCorner.forEach((p, c) => {
                for (let j = 0; j < Math.min(4, elementSize); j++) {
                    const w = weights[p * elementSize + j] ?? 0;
                    if (w <= 0) continue;
                    const joint = indices[p * elementSize + j] ?? 0;
                    boneIDs[c * 4 + j] = remap ? remap[joint]! : joint;
                    boneWeights[c * 4 + j] = w;
                }
                // Normalize in case the sum isn't 1.
                const sum = boneWeights[c * 4]! + boneWeights[c * 4 + 1]! + boneWeights[c * 4 + 2]! + boneWeights[c * 4 + 3]!;
                for (let j = 0; j < 4; j++) boneWeights[c * 4 + j]! /= sum;
            });
            return { boneNodeIDs: ids, inverseBind: skel.bind.map((b) => inverse(b)), boneIDs, weights: boneWeights };
        };
        // Time-sampled xforms (createAnimation): meshes below one get a node chain from the stage root.
        const xformAnims = layerText ? extractUsdXformAnimations(layerText) : new Map<string, UsdaXformAnimation>();
        const nodes: SceneNode[] = [];
        const animations: AnimationChannel[] = [];
        const nodeIds = new Map<string, number>();
        const nodeFor = (path: string): number => {
            const existing = nodeIds.get(path);
            if (existing !== undefined) return existing;
            const parent = path === "" ? -1 : nodeFor(path.slice(0, path.lastIndexOf("/")));
            const n = path === "" ? undefined : nodesByPath.get(path);
            const local = path === "" ? rootXform : n?.localMatrix?.length === 16 ? usdToWebMatrix(n.localMatrix) : float4x4.identity();
            const id = nodes.length;
            nodes.push({ parent, ...decomposeTRS(local) });
            const anim = xformAnims.get(path);
            if (anim) {
                const clip = new Set(animations.map((c) => c.clip)).size;
                const times = Float32Array.from(anim.times);
                const channel = (p: AnimationChannel["path"], values: number[]) => animations.push({ nodeID: id, path: p, times, values: Float32Array.from(values), interp: "LINEAR", clip });
                channel("translation", anim.translation.flatMap((v) => [v.x, v.y, v.z]));
                channel("rotation", anim.rotation.flatMap((q) => [q.x, q.y, q.z, q.w]));
                channel("scale", anim.scaling.flatMap((v) => [v.x, v.y, v.z]));
            }
            nodeIds.set(path, id);
            return id;
        };
        const isAnimated = (path: string | undefined) => {
            for (let p = path ?? ""; p !== ""; p = p.slice(0, p.lastIndexOf("/"))) if (xformAnims.has(p)) return true;
            return false;
        };
        // Stage bounds in USD space (UsdGeomBBoxCache's world bound, without the root transform).
        const lo = [Infinity, Infinity, Infinity];
        const hi = [-Infinity, -Infinity, -Infinity];
        // Animated point instances: prototype subtrees become node chains below each instance's node.
        const walk = (node: UsdNode, parentWorld: float4x4, parentUsd: float4x4, instanced = false, parentNode?: number): void => {
            const instancer = node.absPath ? instancers.get(node.absPath) : undefined;
            if (instancer) {
                const anim = instancer.animation;
                const baseNode = anim ? nodes.push({ parent: -1, ...decomposeTRS(mulMat(rootXform, instancer.usdWorld)) }) - 1 : undefined;
                instancer.instances.forEach(({ proto, transform }, i) => {
                    const protoPath = instancer.prototypes[proto];
                    const protoNode = protoPath ? nodesByPath.get(protoPath) : undefined;
                    if (!protoNode) {
                        Logger.error(`Point instancer '${instancer.path}' references nonexistent prim '${protoPath}'. Ignoring.`);
                        return;
                    }
                    let instanceNode: number | undefined;
                    if (anim) {
                        // One animation per instance on its root node (createPointInstanceKeyframes).
                        instanceNode = nodes.push({ parent: baseNode!, ...decomposeTRS(transform) }) - 1;
                        const clip = new Set(animations.map((c) => c.clip)).size;
                        const times = Float32Array.from(anim.times.map((t) => t / tcps));
                        const trs = anim.transforms.map((frame) => decomposeTRS(frame[i]!));
                        const channel = (path: AnimationChannel["path"], values: number[]) => animations.push({ nodeID: instanceNode!, path, times, values: Float32Array.from(values), interp: "LINEAR", clip });
                        channel("translation", trs.flatMap((k) => [k.t.x, k.t.y, k.t.z]));
                        channel("rotation", trs.flatMap((k) => [k.r.x, k.r.y, k.r.z, k.r.w]));
                        channel("scale", trs.flatMap((k) => [k.s.x, k.s.y, k.s.z]));
                    }
                    const usdWorld = mulMat(instancer.usdWorld, transform);
                    walk(protoNode, mulMat(rootXform, usdWorld), usdWorld, true, instanceNode);
                });
                return;
            }
            let world = parentWorld;
            let usdWorld = parentUsd;
            const ownNode = parentNode !== undefined ? nodes.push({ parent: parentNode, ...decomposeTRS(node.localMatrix?.length === 16 ? usdToWebMatrix(node.localMatrix) : float4x4.identity()) }) - 1 : undefined;
            const meshNodeID = () => ownNode ?? (!instanced && isAnimated(node.absPath) ? nodeFor(node.absPath!) : undefined);
            if (node.localMatrix && node.localMatrix.length === 16) {
                world = mulMat(parentWorld, usdToWebMatrix(node.localMatrix));
                usdWorld = mulMat(parentUsd, usdToWebMatrix(node.localMatrix));
            }
            if (node.nodeType === "mesh") {
                const points = usd.getMesh(node.contentId).points;
                for (let i = 0; i < points.length; i += 3) {
                    const p = transformPoint(usdWorld, new float3(points[i]!, points[i + 1]!, points[i + 2]!));
                    [p.x, p.y, p.z].forEach((v, k) => {
                        lo[k] = Math.min(lo[k]!, v);
                        hi[k] = Math.max(hi[k]!, v);
                    });
                }
            }
            if (node.nodeType === "mesh" && !excludePrims?.has(node.primName)) {
                const mesh = usd.getMesh(node.contentId);
                const level = node.absPath ? refinementLevels.get(node.absPath) : undefined;
                const usdaMesh = node.absPath ? usdaMeshes.get(node.absPath) : undefined;
                const samples = usdaMesh?.pointsSamples;
                // UsdSkel: skinned by its bound skeleton when that has an animation, as natively.
                const skel = usdaMesh?.skin && node.absPath ? skeletons.get(skelBindings.get(node.absPath) ?? "") : undefined;
                if (usdaMesh?.skin && !skel?.anim) {
                    Logger.warning(skel ? `SkelRoot contains a skeleton '${skel.path}' without an associated animation, which is not supported. Ignoring.` : `Mesh '${node.absPath}' has skinning data but no skeleton. Skinning data will not be loaded.`);
                } else if (usdaMesh?.skin && skel && usdaMesh.skin.interpolation !== "vertex") {
                    Logger.warning(`Skinning data for mesh '${node.absPath}' must be per-vertex. "constant" interpolation is not supported. Ignoring primitive.`);
                    return;
                } else if (usdaMesh?.skin && skel && !instanced) {
                    const materialID = getOrAddMaterial(mesh.materialId, bindings.get(node.absPath!), node.absPath);
                    const corners = triangulateUsdMesh(usdaMesh, usdaMesh.points, usdaMesh.normals?.values);
                    const vertices = cornerVertices(corners, texCoordTransforms[materialID]!);
                    const indices = Uint32Array.from(vertices.keys());
                    generateTangents(vertices, indices);
                    meshes.push({ vertices, indices, materialID, transform: world.clone(), skin: usdSkin(usdaMesh, skel, corners.pointIndices!, world, node.absPath!) });
                    for (const child of node.children ?? []) walk(child, world, usdWorld, instanced, ownNode);
                    return;
                }
                const motion = samples && samples.length > 1 && options.settings?.getAttribute(node.absPath!, "usdImporter:enableMotion", 1) !== false && options.settings?.getAttribute(node.absPath!, "usdImporter:enableMotion", 1) !== 0;
                if (motion && usdaMesh) {
                    // Time-sampled points: a vertex cache (CachedMesh), each sample converted like the base mesh.
                    const materialID = getOrAddMaterial(mesh.materialId, bindings.get(node.absPath!), node.absPath);
                    const toTexCrd = texCoordTransforms[materialID]!;
                    const frames = samples.map((s) => {
                        const normals = usdaMesh.normalsSamples ? sampleAt(usdaMesh.normalsSamples, s.time) : usdaMesh.normals?.values;
                        const refinedSample = level && osd ? tessellateUsdMesh(osd, node.absPath!, { ...usdaMesh, points: s.value }, level) : null;
                        const corners = refinedSample ? refinedCorners(refinedSample) : triangulateUsdMesh(usdaMesh, s.value, normals);
                        const vertices = cornerVertices(corners, toTexCrd);
                        generateTangents(vertices, Uint32Array.from(vertices.keys()));
                        return vertices;
                    });
                                meshes.push({
                        vertices: frames[0]!,
                        indices: Uint32Array.from(frames[0]!.keys()),
                        materialID,
                        transform: world.clone(),
                        nodeID: meshNodeID(),
                        vertexCache: { times: samples.map((s) => s.time / tcps), frames },
                    });
                    for (const child of node.children ?? []) walk(child, world, usdWorld, instanced, ownNode);
                    return;
                }
                const refined = level && osd ? tessellateUsdMesh(osd, node.absPath!, usdaMeshes.get(node.absPath!)!, level) : null;
                if (refined) {
                    const materialID = getOrAddMaterial(mesh.materialId, bindings.get(node.absPath!), node.absPath);
                    const { vertices, indices } = refinedVertices(refined, texCoordTransforms[materialID]!);
                    generateTangents(vertices, indices);
                    meshes.push({ vertices, indices, materialID, transform: world.clone(), nodeID: meshNodeID() });
                    for (const child of node.children ?? []) walk(child, world, usdWorld, instanced, ownNode);
                    return;
                }
                let positions: Float32Array = mesh.points;
                let indices: Uint32Array = new Uint32Array(mesh.faceVertexIndices);
                let normals = mesh.normals && mesh.normals.length === positions.length && usdaMesh?.hasNormals !== false ? mesh.normals : null;
                let uvs = mesh.texcoords;
                if (!normals) {
                    // No authored normals: native's triangulate() generates flat per-face normals.
                    const flat = flatNormals(positions, indices, uvs, usdaMesh);
                    ({ positions, indices, normals, uvs } = flat);
                }
                const vertexCount = positions.length / 3;
                const vertices: StaticVertex[] = new Array(vertexCount);
                const materialID = getOrAddMaterial(mesh.materialId, node.absPath ? bindings.get(node.absPath) : undefined, node.absPath);
                const toTexCrd = texCoordTransforms[materialID]!;
                for (let i = 0; i < vertexCount; i++) {
                    vertices[i] = {
                        position: new float3(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!),
                        normal: new float3(normals[i * 3]!, normals[i * 3 + 1]!, normals[i * 3 + 2]!),
                        tangent: new float4(0, 0, 0, 0),
                        // The material's texcoord transform (V flip for UsdPreviewSurface), as natively.
                        texCrd: uvs && uvs.length === vertexCount * 2 ? new float2(...toTexCrd(uvs[i * 2]!, uvs[i * 2 + 1]!)) : new float2(0, 0),
                    };
                }
                generateTangents(vertices, indices);
                meshes.push({ vertices, indices, materialID, transform: world.clone(), nodeID: meshNodeID() });
            } else if (node.nodeType !== "xform" && node.nodeType !== "" && !/camera|light/i.test(node.nodeType)) {
                Logger.warning(`UsdImporter: prim type '${node.nodeType}' ('${node.primName}') not supported (skipped)`);
            }
            for (const child of node.children ?? []) walk(child, world, usdWorld, instanced, ownNode);
        };

        // Native's stage root transform: meters per unit, Z-up rotated to Y-up.
        walk(usd.getDefaultRootNode(), rootXform, float4x4.identity());
        const mpu = stageInfo?.metersPerUnit ?? 1;
        const diagonal = Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!);
        const stage = Number.isFinite(diagonal) ? { center: new float3(((lo[0]! + hi[0]!) / 2) * mpu, ((lo[1]! + hi[1]!) / 2) * mpu, ((lo[2]! + hi[2]!) / 2) * mpu), diagonal: diagonal * mpu } : null;

        // Resolve UsdUVTexture images (URI, embedded-encoded, or pre-decoded).
        if (textureManager) {
            for (const { desc, material: m, inputs } of textureJobs) {
                await resolveMaterialTextures(usd, m, desc, textureManager, baseUrl, !!options.assumeLinearSpaceTextures, inputs);
            }
        }
        const metadata = layerText ? (usdRenderSettings(layerText)?.metadata ?? null) : null;
        return { meshes, materials, materialNames, ...extracted, stage, metadata, nodes, animations };
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
 *  baseColor sRGB unless sourceColorSpace says otherwise; roughness+metallic
 *  packed into one ORM texture like the native CreateSpecularTexture kernel,
 *  reading the connected output channel; opacity through packBaseColorAlpha or
 *  createSpecularTransmissionTexture; normal, emissive and displacement direct.
 *  The packing kernels run on the CPU here (the port decodes textures there). */
async function resolveMaterialTextures(
    usd: TinyUsdzScene,
    m: UsdMaterial,
    desc: SceneMaterialDesc,
    textureManager: TextureManager,
    baseUrl: string,
    assumeLinear = false,
    inputs = new Map<string, UsdaTextureInput>(),
): Promise<void> {
    const valid = (id: number | undefined): id is number => id !== undefined && id >= 0;
    // Authored color spaces override the slot's (ConvertedInput::loadSRGB).
    const srgbOf = (input: string, slotSrgb: boolean) => (inputs.get(input)?.srgb ?? slotSrgb) && !assumeLinear;
    // Single-channel inputs read the connected output (red without layer text); -2 marks several channels.
    const channelOf = (input: string) => {
        const output = inputs.get(input)?.output ?? "r";
        const c = usdChannelIndex(output);
        return c >= 0 ? c : -2;
    };
    const load = async (id: number, input: string, slotSrgb: boolean): Promise<number | undefined> => {
        try {
            return textureManager.addTexture({ bitmap: await resolveImageBitmap(usd, id, baseUrl), srgb: srgbOf(input, slotSrgb) });
        } catch (err) {
            Logger.warning(`UsdImporter: failed to load texture ${id} (${String(err)})`);
            return undefined;
        }
    };
    const threshold = m.opacityThreshold ?? 0;
    const opacityTextured = valid(m.opacityTextureId);
    const opacityChannel = channelOf("opacity");
    if (opacityTextured && opacityChannel < 0) {
        Logger.warning(threshold > 0 ? "Cannot set alpha channel; opacity texture provides more than one channel." : "Cannot create transmission texture; opacity texture provides more than one channel of data.");
    }
    if (threshold > 0 && (valid(m.diffuseColorTextureId) || opacityTextured) && ((m.opacity ?? 1) < 1 || opacityTextured) && !(opacityTextured && opacityChannel < 0)) {
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
                    packed[i + 3] = opacityImg ? sampleNearest(opacityImg, x, y, w, h, opacityChannel) : opacityConst;
                }
            }
            const bitmap = await createImageBitmap(new ImageData(packed, w, h), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
            desc.basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, textureManager.addTexture({ bitmap, srgb: srgbOf("diffusecolor", true) }));
        } catch (err) {
            Logger.warning(`UsdImporter: failed to pack base colour and opacity (${String(err)})`);
        }
    } else if (valid(m.diffuseColorTextureId) && !(threshold > 0 && opacityTextured)) {
        const id = await load(m.diffuseColorTextureId, "diffusecolor", true);
        if (id !== undefined) desc.basic.texBaseColor = packTextureHandle(TextureHandleMode.Texture, id);
    }
    if (threshold <= 0 && opacityTextured && opacityChannel >= 0) {
        // createSpecularTransmissionTexture: textured opacity becomes a grey
        // transmission map of 1 - opacity, with full specular transmission.
        try {
            const opacityImg = readPixels(await resolveImageBitmap(usd, m.opacityTextureId!, baseUrl));
            const out = new Uint8ClampedArray(opacityImg.width * opacityImg.height * 4);
            for (let i = 0; i < out.length; i += 4) {
                const v = 255 - opacityImg.data[i + opacityChannel]!;
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
        const id = await load(m.displacementTextureId, "displacement", false);
        if (id !== undefined) desc.basic.texDisplacement = packTextureHandle(TextureHandleMode.Texture, id);
    }
    const roughChannel = channelOf("roughness");
    const metalChannel = channelOf("metallic");
    if (valid(m.roughnessTextureId) && roughChannel < 0) Logger.warning("Cannot create specular texture; roughness texture provides more than one channel.");
    else if (valid(m.metallicTextureId) && metalChannel < 0) Logger.warning("Cannot create specular texture; metallic texture provides more than one channel.");
    else if (valid(m.roughnessTextureId) || valid(m.metallicTextureId)) {
        try {
            const rough = valid(m.roughnessTextureId) ? readPixels(await resolveImageBitmap(usd, m.roughnessTextureId, baseUrl)) : null;
            const metal = valid(m.metallicTextureId) ? readPixels(await resolveImageBitmap(usd, m.metallicTextureId, baseUrl)) : null;
            const w = Math.max(rough?.width ?? 0, metal?.width ?? 0);
            const h = Math.max(rough?.height ?? 0, metal?.height ?? 0);
            const orm = new Uint8ClampedArray(w * h * 4);
            const roughConst = Math.round((desc.basic.specular?.y ?? 0.5) * 255);
            const metalConst = Math.round((desc.basic.specular?.z ?? 0) * 255);
            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const i = (y * w + x) * 4;
                    orm[i + 1] = rough ? sampleNearest(rough, x, y, w, h, roughChannel) : roughConst;
                    orm[i + 2] = metal ? sampleNearest(metal, x, y, w, h, metalChannel) : metalConst;
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
        const id = await load(m.normalTextureId, "normal", false);
        if (id !== undefined) desc.basic.texNormalMap = packTextureHandle(TextureHandleMode.Texture, id);
    }
    if (valid(m.emissiveColorTextureId)) {
        const id = await load(m.emissiveColorTextureId, "emissivecolor", true);
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
