/**
 * glTF 2.0 importer (Scene/Importer) — the web-native scene format
 * (native Falcor loads the same files via Assimp, enabling oracle comparison).
 *
 * v1 scope: static triangle meshes (POSITION/NORMAL/TANGENT/TEXCOORD_0,
 * u16/u32 indices), node-hierarchy transforms, pbrMetallicRoughness factors.
 * Textures land with the TextureManager packing work; skinning in M7.
 */

import type { Device } from "../../Core/API/Device.js";
import { Scene, type SceneMaterialDesc, type SceneMeshDesc } from "../Scene.js";
import { float2, float3, float4, normalize3 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat, matrixFromTranslation, matrixFromScaling, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { matrixFromQuat, quatf } from "../../Utils/Math/Quaternion.js";
import { RuntimeError } from "../../Core/Error.js";
import { Logger } from "../../Utils/Logger.js";
import { LightType, createPackedVertices, kPackedVertexFloats, type AnalyticLight, type StaticVertex } from "../SceneData.js";
import { decomposeTRS, type SceneNode, type AnimationChannel, type AnimationPath, type SkinDesc, type MorphDesc, type WeightTrack } from "../Animation/SceneAnimation.js";
import { TextureManager } from "../Material/TextureManager.js";
import { TextureHandleMode, packTextureHandle } from "../Material/MaterialData.js";
import { fovYToFocalLength } from "../Camera/Camera.js";

interface GltfLight {
    type: "point" | "directional" | "spot";
    color?: number[];
    intensity?: number;
    range?: number;
    spot?: { innerConeAngle?: number; outerConeAngle?: number };
}

interface GltfJson {
    asset: { version: string };
    scene?: number;
    scenes?: { nodes: number[] }[];
    nodes?: {
        mesh?: number;
        skin?: number;
        camera?: number;
        children?: number[];
        matrix?: number[];
        translation?: number[];
        rotation?: number[];
        scale?: number[];
        weights?: number[];
        extensions?: { KHR_lights_punctual?: { light: number } };
    }[];
    extensions?: { KHR_lights_punctual?: { lights: GltfLight[] } };
    skins?: { joints: number[]; inverseBindMatrices?: number }[];
    animations?: {
        channels: { target: { node?: number; path: string }; sampler: number }[];
        samplers: { input: number; output: number; interpolation?: string }[];
    }[];
    meshes?: { primitives: GltfPrimitive[]; weights?: number[] }[];
    accessors?: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string; normalized?: boolean }[];
    bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
    buffers?: { uri?: string; byteLength: number }[];
    materials?: {
        name?: string;
        pbrMetallicRoughness?: {
            baseColorFactor?: number[];
            metallicFactor?: number;
            roughnessFactor?: number;
            baseColorTexture?: { index: number; extensions?: { KHR_texture_transform?: { offset?: number[]; scale?: number[]; rotation?: number } } };
        };
        emissiveFactor?: number[];
        doubleSided?: boolean;
    }[];
    textures?: { source?: number; sampler?: number }[];
    images?: { uri?: string; bufferView?: number; mimeType?: string }[];
    cameras?: { type?: string; perspective?: { yfov?: number; aspectRatio?: number; znear?: number } }[];
}

/** Camera pose extracted from a glTF camera node (bind pose). */
export interface GltfCameraPose {
    position: float3;
    target: float3;
    up: float3;
    focalLength: number;
    /** Near/far planes, when the importer provides them. */
    depthRange?: [number, number];
    /** Depth of field and film size, when the importer provides them (USD cameras). */
    focalDistance?: number;
    apertureRadius?: number;
    frameWidth?: number;
    frameHeight?: number;
    name?: string;
}

interface GltfPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    material?: number;
    mode?: number;
    targets?: Record<string, number>[]; // morph targets (POSITION/NORMAL deltas)
    extensions?: { KHR_draco_mesh_compression?: { bufferView: number; attributes: Record<string, number> } };
}

/** Decodes a primitive's `KHR_draco_mesh_compression` payload into plain arrays. */
async function decodeDracoPrimitive(
    json: { bufferViews?: { buffer: number; byteOffset?: number; byteLength: number }[] },
    buffers: Uint8Array[],
    ext: { bufferView: number; attributes: Record<string, number> },
) {
    const { decodeDracoMesh } = await import("./DracoDecoder.js");
    const view = json.bufferViews![ext.bufferView]!;
    const buffer = buffers[view.buffer]!;
    const start = buffer.byteOffset + (view.byteOffset ?? 0);
    return decodeDracoMesh(new Uint8Array(buffer.buffer, start, view.byteLength), ext.attributes);
}

const kComponentSize: Record<number, number> = { 5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4 };
const kTypeComponents: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

export class GltfImporter {
    /** Imports a .gltf (JSON, embedded/external buffers) or .glb from a URL. */
    static async importFromUrl(device: Device, url: string, lights: AnalyticLight[] = []): Promise<Scene> {
        const response = await fetch(url);
        if (!response.ok) throw new RuntimeError(`GltfImporter: failed to fetch '${url}' (${response.status})`);
        const bytes = new Uint8Array(await response.arrayBuffer());
        return GltfImporter.importFromBytes(device, bytes, url, lights);
    }

    static async importFromBytes(device: Device, bytes: Uint8Array, baseUrl = "", lights: AnalyticLight[] = []): Promise<Scene> {
        const textureManager = new TextureManager();
        const parsed = await GltfImporter.parseToDescs(bytes, baseUrl, textureManager);
        const scene = new Scene(device, parsed.meshes, parsed.materials, [...lights, ...parsed.lights], textureManager, [], parsed.nodes, parsed.animations, parsed.cameraNodeID, parsed.weightTracks);
        if (parsed.camera) {
            scene.camera.setPosition(parsed.camera.position);
            scene.camera.setTarget(parsed.camera.target);
            scene.camera.setUpVector(parsed.camera.up);
            scene.camera.setFocalLength(parsed.camera.focalLength);
        }
        return scene;
    }

    /** Parses glTF into scene descriptors without constructing GPU resources
     *  (shared by importFromBytes and the pyscene SceneBuilder bridge). */
    static async parseToDescs(
        bytes: Uint8Array,
        baseUrl = "",
        textureManager = new TextureManager(),
        options: { assumeLinearSpaceTextures?: boolean; useOriginalTangentSpace?: boolean } = {},
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; nodes: SceneNode[]; animations: AnimationChannel[]; lights: AnalyticLight[]; cameraNodeID?: number; camera?: GltfCameraPose; weightTracks: WeightTrack[] }> {
        let json: GltfJson;
        let binChunk: Uint8Array | null = null;

        if (bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {
            // GLB container: 12-byte header, then chunks (JSON, BIN).
            const dv = new DataView(bytes.buffer, bytes.byteOffset);
            let offset = 12;
            let jsonText = "";
            while (offset < bytes.byteLength) {
                const chunkLength = dv.getUint32(offset, true);
                const chunkType = dv.getUint32(offset + 4, true);
                const chunk = bytes.subarray(offset + 8, offset + 8 + chunkLength);
                if (chunkType === 0x4e4f534a) jsonText = new TextDecoder().decode(chunk);
                else if (chunkType === 0x004e4942) binChunk = chunk;
                offset += 8 + chunkLength + ((4 - (chunkLength % 4)) % 4);
            }
            json = JSON.parse(jsonText);
        } else {
            json = JSON.parse(new TextDecoder().decode(bytes));
        }

        // Resolve buffers (GLB bin chunk, data: URIs, or relative fetches).
        const buffers: Uint8Array[] = [];
        for (const buf of json.buffers ?? []) {
            if (!buf.uri) {
                if (!binChunk) throw new RuntimeError("GltfImporter: buffer without uri and no GLB bin chunk");
                buffers.push(binChunk);
            } else if (buf.uri.startsWith("data:")) {
                const base64 = buf.uri.slice(buf.uri.indexOf(",") + 1);
                const bin = atob(base64);
                const out = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
                buffers.push(out);
            } else {
                const bufUrl = new URL(buf.uri, new URL(baseUrl, "http://x/")).pathname;
                const res = await fetch(bufUrl);
                if (!res.ok) throw new RuntimeError(`GltfImporter: failed to fetch buffer '${bufUrl}'`);
                buffers.push(new Uint8Array(await res.arrayBuffer()));
            }
        }

        /** Quantized attributes arrive as integers; downstream code wants floats. */
        const toFloats = (data: Float32Array | Uint32Array): Float32Array => (data instanceof Float32Array ? data : Float32Array.from(data));

        const readAccessor = (index: number): Float32Array | Uint32Array => {
            const acc = json.accessors![index]!;
            const view = json.bufferViews![acc.bufferView!]!;
            const buffer = buffers[view.buffer]!;
            const components = kTypeComponents[acc.type]!;
            const compSize = kComponentSize[acc.componentType]!;
            const stride = view.byteStride ?? components * compSize;
            const base = buffer.byteOffset + (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);

            const dv = new DataView(buffer.buffer);
            // Reads one component in its declared type. Signed types matter for
            // KHR_mesh_quantization, which stores normals/uvs as byte/short.
            const component = (at: number): number => {
                switch (acc.componentType) {
                    case 5126: return dv.getFloat32(at, true); // float
                    case 5125: return dv.getUint32(at, true); // unsigned int
                    case 5123: return dv.getUint16(at, true); // unsigned short
                    case 5122: return dv.getInt16(at, true); // short
                    case 5121: return dv.getUint8(at); // unsigned byte
                    case 5120: return dv.getInt8(at); // byte
                    default: throw new RuntimeError(`glTF: unsupported componentType ${acc.componentType}`);
                }
            };
            // Normalized integers map onto [0,1] or [-1,1] (glTF 3.11 / KHR_mesh_quantization).
            const normalize = (v: number): number => {
                switch (acc.componentType) {
                    case 5121: return v / 255;
                    case 5123: return v / 65535;
                    case 5120: return Math.max(v / 127, -1);
                    case 5122: return Math.max(v / 32767, -1);
                    default: return v;
                }
            };

            if (acc.componentType === 5126 || acc.normalized) {
                const out = new Float32Array(acc.count * components);
                for (let i = 0; i < acc.count; i++) {
                    for (let c = 0; c < components; c++) {
                        const raw = component(base + i * stride + c * compSize);
                        out[i * components + c] = acc.normalized ? normalize(raw) : raw;
                    }
                }
                return out;
            }
            const out = new Uint32Array(acc.count * components);
            for (let i = 0; i < acc.count; i++) {
                for (let c = 0; c < components; c++) out[i * components + c] = component(base + i * stride + c * compSize);
            }
            return out;
        };

        // Decode images -> TextureManager (baseColor textures are sRGB).
        // Each image is an independent fetch + decode, so they run concurrently;
        // registration stays in declaration order, which keeps texture IDs (and
        // the material handles built from them) independent of completion order.
        const textureIDs = new Map<number, number>();
        const decodeTexture = async (t: number): Promise<ImageBitmap | null> => {
            const tex = json.textures![t]!;
            if (tex.source === undefined) return null;
            const img = json.images![tex.source]!;
            let blob: Blob;
            if (img.uri?.startsWith("data:")) {
                const b64 = img.uri.slice(img.uri.indexOf(",") + 1);
                const bin = atob(b64);
                const bytesArr = new Uint8Array(bin.length);
                for (let i = 0; i < bin.length; i++) bytesArr[i] = bin.charCodeAt(i);
                blob = new Blob([bytesArr], { type: img.mimeType ?? "image/png" });
            } else if (img.bufferView !== undefined) {
                const view = json.bufferViews![img.bufferView]!;
                const buf = buffers[view.buffer]!;
                blob = new Blob([buf.slice(view.byteOffset ?? 0, (view.byteOffset ?? 0) + view.byteLength) as Uint8Array<ArrayBuffer>], { type: img.mimeType ?? "image/png" });
            } else {
                const imgUrl = new URL(img.uri!, new URL(baseUrl, "http://x/")).pathname;
                blob = await (await fetch(imgUrl)).blob();
            }
            return createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
        };
        const decoded = await Promise.all(Array.from({ length: (json.textures ?? []).length }, (_v, t) => decodeTexture(t)));
        decoded.forEach((bitmap, t) => {
            if (bitmap) textureIDs.set(t, textureManager.addTexture({ bitmap, srgb: !options.assumeLinearSpaceTextures }));
        });

        // Materials (pbrMetallicRoughness factors; MetalRough encoding: specular = (occlusion, roughness, metallic)).
        const materials: SceneMaterialDesc[] = (json.materials ?? []).map((m) => {
            const pbr = m.pbrMetallicRoughness ?? {};
            const bc = pbr.baseColorFactor ?? [1, 1, 1, 1];
            const emissive = m.emissiveFactor ?? [0, 0, 0];
            const baseColorTex = pbr.baseColorTexture !== undefined ? textureIDs.get(pbr.baseColorTexture.index) : undefined;
            return {
                ...(m.name !== undefined ? { name: m.name } : {}),
                // Emissive flag mirrors BasicMaterial::updateEmissiveFlag (factor defaults to 1).
                header: { doubleSided: m.doubleSided ?? false, emissive: emissive.some((c) => c !== 0) },
                basic: {
                    baseColor: new float4(bc[0]!, bc[1]!, bc[2]!, bc[3]!),
                    // specular.r stays 0 as natively (Assimp's glTF path sets roughness and metallic only).
                    specular: new float4(0, pbr.roughnessFactor ?? 1, pbr.metallicFactor ?? 1, 0),
                    emissive: new float3(emissive[0]!, emissive[1]!, emissive[2]!),
                    texBaseColor: baseColorTex !== undefined ? packTextureHandle(TextureHandleMode.Texture, baseColorTex) : undefined,
                },
            };
        });
        if (materials.length === 0) materials.push({ basic: { baseColor: new float4(1, 1, 1, 1) } });

        // Retained node graph (indexed by glTF node index) for animation: parent
        // links from children lists, plus each node's bind-pose local TRS.
        const gltfNodes = json.nodes ?? [];
        const parentOf = new Array<number>(gltfNodes.length).fill(-1);
        gltfNodes.forEach((nd, i) => (nd.children ?? []).forEach((c) => (parentOf[c] = i)));
        const sceneNodes: SceneNode[] = gltfNodes.map((nd, i) => {
            if (nd.matrix) {
                const m = new float4x4();
                for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) m.set(r, c, nd.matrix[c * 4 + r]!);
                return { parent: parentOf[i]!, ...decomposeTRS(m) };
            }
            const t = nd.translation ?? [0, 0, 0];
            const r = nd.rotation ?? [0, 0, 0, 1];
            const s = nd.scale ?? [1, 1, 1];
            return { parent: parentOf[i]!, t: new float3(t[0]!, t[1]!, t[2]!), r: new quatf(r[0]!, r[1]!, r[2]!, r[3]!), s: new float3(s[0]!, s[1]!, s[2]!) };
        });

        // Animation channels (translation/rotation/scale keyframe tracks per node)
        // plus morph "weights" tracks (numTargets weights per keyframe).
        const animations: AnimationChannel[] = [];
        const weightTracks: WeightTrack[] = [];
        for (const anim of json.animations ?? []) {
            for (const ch of anim.channels) {
                const path = ch.target.path;
                if (ch.target.node === undefined) continue;
                const sampler = anim.samplers[ch.sampler]!;
                const interp = sampler.interpolation === "STEP" ? "STEP" : sampler.interpolation === "CUBICSPLINE" ? "CUBICSPLINE" : "LINEAR";
                if (path === "translation" || path === "rotation" || path === "scale") {
                    animations.push({ nodeID: ch.target.node, path: path as AnimationPath, times: readAccessor(sampler.input) as Float32Array, values: readAccessor(sampler.output) as Float32Array, interp });
                } else if (path === "weights") {
                    const meshIdx = gltfNodes[ch.target.node]?.mesh;
                    const numTargets = meshIdx !== undefined ? (json.meshes![meshIdx]!.primitives[0]?.targets?.length ?? 0) : 0;
                    if (numTargets > 0) {
                        weightTracks.push({ nodeID: ch.target.node, times: readAccessor(sampler.input) as Float32Array, values: readAccessor(sampler.output) as Float32Array, numTargets, interp });
                    }
                }
            }
        }

        // Flatten the node hierarchy, collecting world transforms per mesh instance.
        const meshDescs: SceneMeshDesc[] = [];
        const nodeTransform = (node: NonNullable<GltfJson["nodes"]>[number]): float4x4 => {
            if (node.matrix) {
                // glTF matrices are column-major.
                const m = new float4x4();
                for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) m.set(r, c, node.matrix[c * 4 + r]!);
                return m;
            }
            const t = node.translation ?? [0, 0, 0];
            const r = node.rotation ?? [0, 0, 0, 1];
            const s = node.scale ?? [1, 1, 1];
            return mulMat(
                matrixFromTranslation(new float3(t[0]!, t[1]!, t[2]!)),
                mulMat(matrixFromQuat(new quatf(r[0]!, r[1]!, r[2]!, r[3]!)), matrixFromScaling(new float3(s[0]!, s[1]!, s[2]!))),
            );
        };

        // KHR_lights_punctual: light shapes defined at document scope, referenced
        // per-node; placed by the node's world transform (glTF lights aim down -Z).
        const lightDefs = json.extensions?.KHR_lights_punctual?.lights ?? [];
        const lights: AnalyticLight[] = [];
        let cameraNodeID: number | undefined;
        let cameraPose: GltfCameraPose | undefined;

        // Async because Draco-compressed primitives decode through a wasm module.
        const visit = async (nodeIndex: number, parent: float4x4): Promise<void> => {
            const node = json.nodes![nodeIndex]!;
            const world = mulMat(parent, nodeTransform(node));
            const lightRef = node.extensions?.KHR_lights_punctual?.light;
            if (lightRef !== undefined && lightDefs[lightRef]) {
                const L = lightDefs[lightRef]!;
                const color = L.color ?? [1, 1, 1];
                const intensity = new float3(color[0]! * (L.intensity ?? 1), color[1]! * (L.intensity ?? 1), color[2]! * (L.intensity ?? 1));
                const dirW = normalize3(transformVector(world, new float3(0, 0, -1)));
                if (L.type === "directional") {
                    lights.push({ type: LightType.Directional, dirW, intensity, nodeID: nodeIndex });
                } else {
                    // point and spot are both Falcor PointLight (spot = cone cutoff).
                    const light: AnalyticLight = { type: LightType.Point, posW: transformPoint(world, new float3(0, 0, 0)), dirW, intensity, nodeID: nodeIndex };
                    if (L.type === "spot") {
                        light.openingAngle = L.spot?.outerConeAngle ?? Math.PI / 4;
                        light.penumbraAngle = Math.max(0, (L.spot?.outerConeAngle ?? Math.PI / 4) - (L.spot?.innerConeAngle ?? 0));
                    }
                    lights.push(light);
                }
            }
            if (node.camera !== undefined && cameraNodeID === undefined) {
                cameraNodeID = nodeIndex;
                const pos = transformPoint(world, new float3(0, 0, 0));
                const fwd = normalize3(transformVector(world, new float3(0, 0, -1)));
                const yfov = json.cameras?.[node.camera]?.perspective?.yfov ?? Math.PI / 4;
                cameraPose = {
                    position: pos,
                    target: new float3(pos.x + fwd.x, pos.y + fwd.y, pos.z + fwd.z),
                    up: normalize3(transformVector(world, new float3(0, 1, 0))),
                    focalLength: fovYToFocalLength(yfov, 24),
                };
            }
            if (node.mesh !== undefined) {
                for (const prim of json.meshes![node.mesh]!.primitives) {
                    if ((prim.mode ?? 4) !== 4) continue; // triangles only
                    // KHR_draco_mesh_compression: the geometry lives in a compressed
                    // buffer view; the accessors only describe the decoded result.
                    const dracoExt = prim.extensions?.KHR_draco_mesh_compression;
                    const draco = dracoExt ? await decodeDracoPrimitive(json, buffers, dracoExt) : null;
                    // KHR_mesh_quantization may store positions as (unsigned) shorts or
                    // bytes; the node's scale/translation puts them back in place.
                    const attribute = (semantic: string): Float32Array | null => {
                        if (draco) return draco.attributes.get(semantic) ?? null;
                        return prim.attributes[semantic] !== undefined ? toFloats(readAccessor(prim.attributes[semantic]!)) : null;
                    };
                    const rawPos = attribute("POSITION");
                    if (!rawPos) continue; // a primitive without positions has no geometry
                    const rawIndices =
                        draco ? draco.indices
                        : prim.indices !== undefined ? new Uint32Array(readAccessor(prim.indices))
                        : Uint32Array.from({ length: rawPos.length / 3 }, (_v, i) => i);
                    // Native imports glTF with Assimp's JoinIdenticalVertices (tangents only compared when kept).
                    const join = joinIdenticalVertices(rawIndices, rawPos, attribute("NORMAL"), attribute("TEXCOORD_0"), options.useOriginalTangentSpace ? attribute("TANGENT") : null);
                    const pick = (a: Float32Array | null, n: number) => a && gather(a, join.source, n);
                    const pos = pick(rawPos, 3)!;
                    const count = pos.length / 3;
                    const normals = pick(attribute("NORMAL"), 3);
                    const tangents = pick(attribute("TANGENT"), 4);
                    const uvs = pick(attribute("TEXCOORD_0"), 2);
                    // KHR_texture_transform: bake the material's uv transform into the
                    // vertices. Quantized assets rely on it to rescale integer uvs, and
                    // a primitive has exactly one material, so baking is exact.
                    const uvTransform = prim.material !== undefined ? json.materials?.[prim.material]?.pbrMetallicRoughness?.baseColorTexture?.extensions?.KHR_texture_transform : undefined;
                    if (uvs && uvTransform) {
                        if (uvTransform.rotation) Logger.warning("GltfImporter: KHR_texture_transform rotation is not supported");
                        const scale = uvTransform.scale ?? [1, 1];
                        const offset = uvTransform.offset ?? [0, 0];
                        for (let i = 0; i < uvs.length; i += 2) {
                            uvs[i] = uvs[i]! * scale[0]! + offset[0]!;
                            uvs[i + 1] = uvs[i + 1]! * scale[1]! + offset[1]!;
                        }
                    }

                    // Packed f32 vertices (glTF's data is f32): large scenes stay within the JS heap.
                    const data = new Float32Array(count * kPackedVertexFloats);
                    for (let i = 0; i < count; i++) {
                        const o = i * kPackedVertexFloats;
                        data[o] = pos[i * 3]!;
                        data[o + 1] = pos[i * 3 + 1]!;
                        data[o + 2] = pos[i * 3 + 2]!;
                        if (normals) [data[o + 3], data[o + 4], data[o + 5]] = [normals[i * 3]!, normals[i * 3 + 1]!, normals[i * 3 + 2]!];
                        else data[o + 5] = 1;
                        if (tangents) [data[o + 6], data[o + 7], data[o + 8], data[o + 9]] = [tangents[i * 4]!, tangents[i * 4 + 1]!, tangents[i * 4 + 2]!, tangents[i * 4 + 3]!];
                        else [data[o + 6], data[o + 9]] = [1, 1];
                        if (uvs) [data[o + 10], data[o + 11]] = [uvs[i * 2]!, uvs[i * 2 + 1]!];
                    }
                    const vertices: StaticVertex[] = createPackedVertices(count, data);
                    const indices = join.indices;

                    // Skinning: per-vertex joints/weights + the skin's joint→node
                    // mapping and inverse-bind matrices (node indices are offset by
                    // the SceneBuilder; boneIDs are local indices into skin.joints).
                    let skin: SkinDesc | undefined;
                    if (node.skin !== undefined && prim.attributes["JOINTS_0"] !== undefined && prim.attributes["WEIGHTS_0"] !== undefined) {
                        const gltfSkin = json.skins![node.skin]!;
                        const ibm = readAccessor(gltfSkin.inverseBindMatrices!) as Float32Array;
                        const inverseBind = gltfSkin.joints.map((_j, ji) => {
                            const m = new float4x4();
                            for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) m.set(r, c, ibm[ji * 16 + c * 4 + r]!);
                            return m;
                        });
                        skin = {
                            boneNodeIDs: gltfSkin.joints.slice(),
                            inverseBind,
                            boneIDs: gather(readAccessor(prim.attributes["JOINTS_0"]) as Uint32Array, join.source, 4),
                            weights: gather(readAccessor(prim.attributes["WEIGHTS_0"]) as Float32Array, join.source, 4),
                        };
                    }
                    // Morph targets: per-target POSITION (and optional NORMAL)
                    // deltas, blended by the node/mesh weights each animated frame.
                    let morph: MorphDesc | undefined;
                    if (prim.targets && prim.targets.length > 0) {
                        const targets = prim.targets.map((t) => ({
                            position: gather(readAccessor(t["POSITION"]!) as Float32Array, join.source, 3),
                            normal: t["NORMAL"] !== undefined ? gather(readAccessor(t["NORMAL"]) as Float32Array, join.source, 3) : undefined,
                        }));
                        const baseWeights = node.weights ?? json.meshes![node.mesh]!.weights ?? new Array(targets.length).fill(0);
                        morph = { targets, nodeID: nodeIndex, baseWeights };
                    }
                    // SceneBuilder generates MikkTSpace tangents unless UseOriginalTangentSpace keeps the asset's.
                    meshDescs.push({ vertices, indices, materialID: prim.material ?? 0, transform: world, nodeID: nodeIndex, skin, morph, tangentSpace: tangents ? "asset" : uvs ? "generate" : "noTexCrds" });
                }
            }
            for (const child of node.children ?? []) await visit(child, world);
        };

        const sceneDef = json.scenes?.[json.scene ?? 0];
        for (const rootNode of sceneDef?.nodes ?? []) await visit(rootNode, float4x4.identity());
        if (meshDescs.length === 0) throw new RuntimeError("GltfImporter: no triangle meshes found");

        return { meshes: meshDescs, materials, nodes: sceneNodes, animations, lights, cameraNodeID, camera: cameraPose, weightTracks };
    }
}

/** Copies the `n`-component elements `source[i]` of `data` in order. */
function gather<T extends Float32Array | Uint32Array>(data: T, source: Uint32Array, n: number): T {
    const out = new (data.constructor as { new (len: number): T })(source.length * n);
    source.forEach((src, i) => out.set(data.subarray(src * n, src * n + n), i * n));
    return out;
}

/**
 * Mirrors Assimp 5.2.5's JoinVerticesProcess: used vertices in order, one per exact position whose normal, uv and
 * tangent lie within 1e-5 of an earlier one's (bone weights are not compared; the first vertex's are kept).
 */
export function joinIdenticalVertices(indices: Uint32Array, pos: Float32Array, normals: Float32Array | null, uvs: Float32Array | null, tangents: Float32Array | null): { indices: Uint32Array; source: Uint32Array } {
    const count = pos.length / 3;
    const used = new Uint8Array(count);
    for (const i of indices) used[i] = 1;
    const kEps2 = Math.fround(1e-5 * 1e-5);
    const f = Math.fround;
    const far = (a: Float32Array | null, n: number, i: number, j: number) => {
        if (!a) return false;
        let d = 0;
        for (let c = 0; c < Math.min(n, 3); c++) d = f(d + f(f(a[i * n + c]! - a[j * n + c]!) ** 2));
        return d > kEps2;
    };
    const buckets = new Map<string, number[]>();
    const remap = new Uint32Array(count);
    const source: number[] = [];
    for (let v = 0; v < count; v++) {
        if (!used[v]) continue;
        const key = `${pos[v * 3]},${pos[v * 3 + 1]},${pos[v * 3 + 2]}`;
        const bucket = buckets.get(key) ?? [];
        buckets.set(key, bucket);
        const hit = bucket.find((u) => !far(pos, 3, v, source[u]!) && !far(normals, 3, v, source[u]!) && !far(uvs, 2, v, source[u]!) && !far(tangents, 4, v, source[u]!));
        if (hit !== undefined) remap[v] = hit;
        else {
            remap[v] = source.length;
            bucket.push(source.length);
            source.push(v);
        }
    }
    return { indices: indices.map((i) => remap[i]!), source: Uint32Array.from(source) };
}
