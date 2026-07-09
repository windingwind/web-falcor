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
import { LightType, type AnalyticLight, type StaticVertex } from "../SceneData.js";
import { decomposeTRS, type SceneNode, type AnimationChannel, type AnimationPath, type SkinDesc } from "../Animation/SceneAnimation.js";
import { TextureManager } from "../Material/TextureManager.js";
import { TextureHandleMode, packTextureHandle } from "../Material/MaterialData.js";
import { generateTangents } from "../TangentSpace.js";
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
        extensions?: { KHR_lights_punctual?: { light: number } };
    }[];
    extensions?: { KHR_lights_punctual?: { lights: GltfLight[] } };
    skins?: { joints: number[]; inverseBindMatrices?: number }[];
    animations?: {
        channels: { target: { node?: number; path: string }; sampler: number }[];
        samplers: { input: number; output: number; interpolation?: string }[];
    }[];
    meshes?: { primitives: GltfPrimitive[] }[];
    accessors?: { bufferView?: number; byteOffset?: number; componentType: number; count: number; type: string }[];
    bufferViews?: { buffer: number; byteOffset?: number; byteLength: number; byteStride?: number }[];
    buffers?: { uri?: string; byteLength: number }[];
    materials?: {
        name?: string;
        pbrMetallicRoughness?: {
            baseColorFactor?: number[];
            metallicFactor?: number;
            roughnessFactor?: number;
            baseColorTexture?: { index: number };
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
}

interface GltfPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    material?: number;
    mode?: number;
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
        const scene = new Scene(device, parsed.meshes, parsed.materials, [...lights, ...parsed.lights], textureManager, [], parsed.nodes, parsed.animations, parsed.cameraNodeID);
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
    ): Promise<{ meshes: SceneMeshDesc[]; materials: SceneMaterialDesc[]; nodes: SceneNode[]; animations: AnimationChannel[]; lights: AnalyticLight[]; cameraNodeID?: number; camera?: GltfCameraPose }> {
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

        const readAccessor = (index: number): Float32Array | Uint32Array => {
            const acc = json.accessors![index]!;
            const view = json.bufferViews![acc.bufferView!]!;
            const buffer = buffers[view.buffer]!;
            const components = kTypeComponents[acc.type]!;
            const compSize = kComponentSize[acc.componentType]!;
            const stride = view.byteStride ?? components * compSize;
            const base = buffer.byteOffset + (view.byteOffset ?? 0) + (acc.byteOffset ?? 0);

            if (acc.componentType === 5126) {
                const out = new Float32Array(acc.count * components);
                const dv = new DataView(buffer.buffer);
                for (let i = 0; i < acc.count; i++) {
                    for (let c = 0; c < components; c++) out[i * components + c] = dv.getFloat32(base + i * stride + c * 4, true);
                }
                return out;
            }
            const out = new Uint32Array(acc.count * components);
            const dv = new DataView(buffer.buffer);
            for (let i = 0; i < acc.count; i++) {
                for (let c = 0; c < components; c++) {
                    out[i * components + c] =
                        compSize === 4 ? dv.getUint32(base + i * stride + c * 4, true)
                        : compSize === 2 ? dv.getUint16(base + i * stride + c * 2, true)
                        : dv.getUint8(base + i * stride + c);
                }
            }
            return out;
        };

        // Decode images -> TextureManager (baseColor textures are sRGB).
        const textureIDs = new Map<number, number>();
        for (let t = 0; t < (json.textures ?? []).length; t++) {
            const tex = json.textures![t]!;
            if (tex.source === undefined) continue;
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
            const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none" });
            textureIDs.set(t, textureManager.addTexture({ bitmap, srgb: true }));
        }

        // Materials (pbrMetallicRoughness factors; MetalRough encoding: specular = (occlusion, roughness, metallic)).
        const materials: SceneMaterialDesc[] = (json.materials ?? []).map((m) => {
            const pbr = m.pbrMetallicRoughness ?? {};
            const bc = pbr.baseColorFactor ?? [1, 1, 1, 1];
            const emissive = m.emissiveFactor ?? [0, 0, 0];
            const baseColorTex = pbr.baseColorTexture !== undefined ? textureIDs.get(pbr.baseColorTexture.index) : undefined;
            return {
                // Emissive flag mirrors BasicMaterial::updateEmissiveFlag (factor defaults to 1).
                header: { doubleSided: m.doubleSided ?? false, emissive: emissive.some((c) => c !== 0) },
                basic: {
                    baseColor: new float4(bc[0]!, bc[1]!, bc[2]!, bc[3]!),
                    specular: new float4(1, pbr.roughnessFactor ?? 1, pbr.metallicFactor ?? 1, 0),
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

        // Animation channels (translation/rotation/scale keyframe tracks per node).
        const animations: AnimationChannel[] = [];
        for (const anim of json.animations ?? []) {
            for (const ch of anim.channels) {
                const path = ch.target.path;
                if (ch.target.node === undefined || (path !== "translation" && path !== "rotation" && path !== "scale")) continue;
                const sampler = anim.samplers[ch.sampler]!;
                animations.push({
                    nodeID: ch.target.node,
                    path: path as AnimationPath,
                    times: readAccessor(sampler.input) as Float32Array,
                    values: readAccessor(sampler.output) as Float32Array,
                    interp: sampler.interpolation === "STEP" ? "STEP" : "LINEAR",
                });
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

        const visit = (nodeIndex: number, parent: float4x4) => {
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
                    const pos = readAccessor(prim.attributes["POSITION"]!) as Float32Array;
                    const count = pos.length / 3;
                    const normals = prim.attributes["NORMAL"] !== undefined ? (readAccessor(prim.attributes["NORMAL"]) as Float32Array) : null;
                    const tangents = prim.attributes["TANGENT"] !== undefined ? (readAccessor(prim.attributes["TANGENT"]) as Float32Array) : null;
                    const uvs = prim.attributes["TEXCOORD_0"] !== undefined ? (readAccessor(prim.attributes["TEXCOORD_0"]) as Float32Array) : null;

                    const vertices: StaticVertex[] = [];
                    for (let i = 0; i < count; i++) {
                        vertices.push({
                            position: new float3(pos[i * 3]!, pos[i * 3 + 1]!, pos[i * 3 + 2]!),
                            normal: normals ? new float3(normals[i * 3]!, normals[i * 3 + 1]!, normals[i * 3 + 2]!) : new float3(0, 0, 1),
                            tangent: tangents
                                ? new float4(tangents[i * 4]!, tangents[i * 4 + 1]!, tangents[i * 4 + 2]!, tangents[i * 4 + 3]!)
                                : new float4(1, 0, 0, 1),
                            texCrd: uvs ? new float2(uvs[i * 2]!, uvs[i * 2 + 1]!) : new float2(0, 0),
                        });
                    }
                    const indices =
                        prim.indices !== undefined
                            ? new Uint32Array(readAccessor(prim.indices))
                            : new Uint32Array(Array.from({ length: count }, (_v, i) => i));
                    // SceneBuilder generates MikkTSpace tangents when the asset has none.
                    if (!tangents) generateTangents(vertices, indices);

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
                            boneIDs: readAccessor(prim.attributes["JOINTS_0"]) as Uint32Array,
                            weights: readAccessor(prim.attributes["WEIGHTS_0"]) as Float32Array,
                        };
                    }
                    meshDescs.push({ vertices, indices, materialID: prim.material ?? 0, transform: world, nodeID: nodeIndex, skin });
                }
            }
            for (const child of node.children ?? []) visit(child, world);
        };

        const sceneDef = json.scenes?.[json.scene ?? 0];
        for (const rootNode of sceneDef?.nodes ?? []) visit(rootNode, float4x4.identity());
        if (meshDescs.length === 0) throw new RuntimeError("GltfImporter: no triangle meshes found");

        return { meshes: meshDescs, materials, nodes: sceneNodes, animations, lights, cameraNodeID, camera: cameraPose };
    }
}
