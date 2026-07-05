/**
 * Scene host class mirroring Falcor/Scene/Scene.h (v1: static triangle meshes,
 * basic materials, single camera; grows toward full parity per milestone).
 *
 * Owns the GPU buffers laid out per SceneTypes.slang and binds the gScene
 * parameter block (upstream Scene.slang via WebFalcor overrides).
 */

import type { Device } from "../Core/API/Device.js";
import { Buffer } from "../Core/API/Buffer.js";
import { Texture } from "../Core/API/Texture.js";
import { Sampler } from "../Core/API/Sampler.js";
import { ResourceBindFlags, MemoryType, ResourceType } from "../Core/API/Types.js";
import { ResourceFormat } from "../Core/API/Formats.js";
import { DefineList } from "../Core/Program/DefineList.js";
import type { ShaderVar } from "../Core/Program/ParameterBlock.js";
import { Camera } from "./Camera/Camera.js";
import { float4x4, transpose, inverse } from "../Utils/Math/Matrix.js";
import {
    GeometryType,
    packGeometryInstances,
    packMeshDescs,
    packStaticVertices,
    type GeometryInstance,
    type MeshDescData,
    type StaticVertex,
} from "./SceneData.js";
import { packBasicMaterialBlob, MaterialType, type BasicMaterialDesc, type MaterialHeaderDesc } from "./Material/MaterialData.js";
import { assert } from "../Core/Error.js";

export interface SceneMeshDesc {
    vertices: StaticVertex[];
    indices: Uint32Array;
    materialID: number;
}

export interface SceneMaterialDesc {
    header?: Partial<MaterialHeaderDesc>;
    basic: BasicMaterialDesc;
}

export class Scene {
    readonly camera = new Camera();
    private buffers: Record<string, Buffer> = {};
    private textureArray: Texture;
    private dummyTexture: Texture;
    private texture3D: Texture;
    private sampler: Sampler;
    private materialCount = 0;
    private instanceCount = 0;
    private drawList: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[] = [];

    constructor(
        public readonly device: Device,
        meshes: SceneMeshDesc[],
        materials: SceneMaterialDesc[],
    ) {
        assert(meshes.length > 0 && materials.length > 0, "Scene requires geometry and materials");

        // Concatenate mesh geometry into global vertex/index buffers.
        const allVertices: StaticVertex[] = [];
        const allIndices: number[] = [];
        const meshDescs: MeshDescData[] = [];
        const instances: GeometryInstance[] = [];
        meshes.forEach((mesh, meshID) => {
            const vbOffset = allVertices.length;
            const ibOffset = allIndices.length;
            allVertices.push(...mesh.vertices);
            for (const i of mesh.indices) allIndices.push(i);
            meshDescs.push({
                vbOffset,
                ibOffset,
                vertexCount: mesh.vertices.length,
                indexCount: mesh.indices.length,
                materialID: mesh.materialID,
            });
            instances.push({
                type: GeometryType.TriangleMesh,
                globalMatrixID: 0,
                materialID: mesh.materialID,
                geometryID: meshID,
                vbOffset,
                ibOffset,
                instanceIndex: meshID,
                geometryIndex: 0,
            });
        });
        this.instanceCount = instances.length;
        this.drawList = meshDescs.map((m, i) => ({
            indexCount: m.indexCount,
            firstIndex: m.ibOffset,
            baseVertex: m.vbOffset,
            firstInstance: i,
        }));

        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const make = (name: string, data: ArrayBufferView | ArrayBuffer, structSize: number, extraFlags = ResourceBindFlags.None) => {
            const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            const buf = new Buffer(this.device, { size: bytes.byteLength, structSize, bindFlags: storage | extraFlags, memoryType: MemoryType.DeviceLocal, name: `Scene::${name}` });
            buf.setBlob(bytes);
            this.buffers[name] = buf;
            return buf;
        };

        make("vertices", packStaticVertices(allVertices), 48, ResourceBindFlags.Vertex);
        make("indices", new Uint32Array(allIndices), 4, ResourceBindFlags.Index);
        make("drawIDs", new Uint32Array(instances.map((inst) => inst.instanceIndex)), 4, ResourceBindFlags.Vertex);
        make("meshes", packMeshDescs(meshDescs), 32);
        make("geometryInstances", packGeometryInstances(instances), 32);

        // Node transforms (identity, single node).
        const identity = float4x4.identity();
        make("worldMatrices", identity.toArray(), 64);
        make("inverseTransposeWorldMatrices", transpose(inverse(identity)).toArray(), 64);

        // Materials.
        this.materialCount = materials.length;
        const blobBytes = new Uint8Array(materials.length * 128);
        materials.forEach((m, i) => {
            blobBytes.set(packBasicMaterialBlob({ materialType: MaterialType.Standard, ...m.header }, m.basic), i * 128);
        });
        make("materialData", blobBytes, 128);
        make("materialBuffer0", new Uint32Array(4), 4);
        make("materialTextureUvScale", new Float32Array([1, 1]), 8);

        // Material system textures (v1: single white layer; TextureManager packing grows here).
        this.textureArray = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            arraySize: 1,
            mipLevels: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "Scene::materialTexturesArray",
        });
        this.textureArray.setSubresourceBlob(0, 0, new Float32Array([1, 1, 1, 1]));
        this.dummyTexture = this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0, 0, 0, 0]));
        this.texture3D = this.device.createTexture3D(1, 1, 1, ResourceFormat.RGBA32Float, 1);
        this.gridRangeTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.RG32Float, 1);
        this.gridIndirectionTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.RGBA32Uint, 1);
        this.gridAtlasTex = this.device.createTexture3D(1, 1, 1, ResourceFormat.R32Float, 1);
        this.sampler = this.device.createSampler();
    }

    private gridRangeTex: Texture;
    private gridIndirectionTex: Texture;
    private gridAtlasTex: Texture;

    /** Mirrors Scene::getSceneDefines(). */
    getSceneDefines(): DefineList {
        return new DefineList().addAll({
            SCENE_GEOMETRY_TYPES: 1 << GeometryType.TriangleMesh,
            SCENE_GRID_COUNT: 0,
            SCENE_SDF_GRID_COUNT: 0,
            SCENE_HAS_INDEXED_VERTICES: 1,
            SCENE_HAS_16BIT_INDICES: 0,
            SCENE_HAS_32BIT_INDICES: 1,
            SCENE_INDEX_BUFFER_COUNT: 1,
            SCENE_INDEX_BUFFER_INDEX_BITS: 1,
            SCENE_VERTEX_BUFFER_COUNT: 1,
            SCENE_VERTEX_BUFFER_INDEX_BITS: 1,
            HIT_INFO_DEFINES: 1,
            HIT_INFO_USE_COMPRESSION: 0,
            HIT_INFO_TYPE_BITS: 4,
            HIT_INFO_INSTANCE_ID_BITS: 16,
            HIT_INFO_PRIMITIVE_INDEX_BITS: 12,
            MATERIAL_SYSTEM_SAMPLER_DESC_COUNT: 16,
            MATERIAL_SYSTEM_TEXTURE_DESC_COUNT: 1,
            MATERIAL_SYSTEM_BUFFER_DESC_COUNT: 1,
            MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT: 1,
            MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED: 0,
            MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS: 0,
            MATERIAL_SYSTEM_USE_LIGHT_PROFILE: 0,
            FALCOR_MATERIAL_INSTANCE_SIZE: 256,
            WEBFALCOR_MTL_STANDARD: 1,
            SCENE_DIFFUSE_ALBEDO_MULTIPLIER: "1.0",
            FALCOR_NVAPI_AVAILABLE: 0,
        });
    }

    /** Mirrors Scene::bindShaderData: fills the gScene parameter block. */
    bindShaderData(root: ShaderVar): void {
        const scene = root["gScene"];

        // Camera (uniforms in the block's default buffer; statically-unused sets no-op).
        const cam = this.camera.getData();
        const c = scene["camera"]["data"];
        c["viewMat"] = cam.viewMat;
        c["projMat"] = cam.projMat;
        c["viewProjMat"] = cam.viewProjMat;
        c["viewProjMatNoJitter"] = cam.viewProjMat;
        c["prevViewProjMatNoJitter"] = cam.viewProjMat; // static scenes: prev == current
        c["invViewProj"] = cam.invViewProj;
        c["posW"] = cam.posW.toArray();
        c["focalLength"] = cam.focalLength;
        c["up"] = cam.up.toArray();
        c["aspectRatio"] = cam.aspectRatio;
        c["target"] = cam.target.toArray();
        c["nearZ"] = cam.nearZ;
        c["cameraU"] = cam.cameraU.toArray();
        c["farZ"] = cam.farZ;
        c["cameraV"] = cam.cameraV.toArray();
        c["jitterX"] = cam.jitterX;
        c["cameraW"] = cam.cameraW.toArray();
        c["jitterY"] = cam.jitterY;

        // Geometry.
        scene["worldMatrices"] = this.buffers["worldMatrices"]!;
        scene["inverseTransposeWorldMatrices"] = this.buffers["inverseTransposeWorldMatrices"]!;
        scene["prevWorldMatrices"] = this.buffers["worldMatrices"]!;
        scene["prevInverseTransposeWorldMatrices"] = this.buffers["inverseTransposeWorldMatrices"]!;
        scene["geometryInstances"] = this.buffers["geometryInstances"]!;
        scene["meshes"] = this.buffers["meshes"]!;
        scene["vertices"]["data0"] = this.buffers["vertices"]!;
        scene["prevVertices"] = this.buffers["vertices"]!;
        scene["indexData"]["data0"] = this.buffers["indices"]!;

        // Env map: dummy black texture (no env light in v1; uniforms stay zeroed).
        scene["envMap"]["envMap"] = this.dummyTexture;
        scene["envMap"]["envSampler"] = this.sampler;

        // Grid volume single instance (dummy; SCENE_GRID_COUNT=0 keeps it unreachable).
        scene["grid0"]["buf"] = this.buffers["materialBuffer0"]!;
        scene["grid0"]["rangeTex"] = this.gridRangeTex;
        scene["grid0"]["indirectionTex"] = this.gridIndirectionTex;
        scene["grid0"]["atlasTex"] = this.gridAtlasTex;

        // Light profile (disabled; dummy bindings).
        scene["materials"]["lightProfile"]["texture"] = this.dummyTexture;
        scene["materials"]["lightProfile"]["sampler"] = this.sampler;

        // Material system.
        const materials = scene["materials"];
        materials["materialCount"] = this.materialCount;
        materials["materialData"] = this.buffers["materialData"]!;
        materials["materialSampler0"] = this.sampler;
        materials["materialTexturesArray"] = this.textureArray;
        materials["materialTextureUvScale"] = this.buffers["materialTextureUvScale"]!;
        materials["webfalcorDummyTexture"] = this.dummyTexture;
        materials["materialBuffer0"] = this.buffers["materialBuffer0"]!;
        materials["materialTexture3D0"] = this.texture3D;
    }

    getGeometryInstanceCount(): number {
        return this.instanceCount;
    }

    /** Raster draw data (mirrors Scene::rasterize): buffers + per-mesh indexed draws. */
    getMeshDrawData(): {
        vertexBuffer: Buffer;
        drawIDBuffer: Buffer;
        indexBuffer: Buffer;
        draws: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[];
    } {
        return {
            vertexBuffer: this.buffers["vertices"]!,
            drawIDBuffer: this.buffers["drawIDs"]!,
            indexBuffer: this.buffers["indices"]!,
            draws: this.drawList,
        };
    }
}
