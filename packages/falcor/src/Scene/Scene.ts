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
import { buildBvh, type BvhTriangle } from "./SoftwareRT/Bvh.js";
import { packLights, type AnalyticLight } from "./SceneData.js";
import { TextureManager } from "./Material/TextureManager.js";
import type { EnvMap } from "./Lights/EnvMap.js";
import { buildLightCollection } from "./Lights/LightCollection.js";
import { decodeNormal2x16Host, type Vec3 } from "../Rendering/Lights/LightBVHTypes.js";
import type { EmissiveTriangleInput } from "../Rendering/Lights/LightBVHBuilder.js";
import { transformPoint } from "../Utils/Math/Matrix.js";
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
    /** World transform (row-major float4x4); identity if omitted. */
    transform?: float4x4;
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
    private textureCount = 1;
    private drawList: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[] = [];

    private lightCount = 0;
    private envMap: EnvMap | null = null;
    private hasEmissiveMaterials = false;
    private materialTypes = new Set<MaterialType>();
    private emissiveTriangleCount = 0;
    private emissiveMeshCount = 0;
    private emissiveFluxes = new Float32Array(0);
    private emissiveTriangles: EmissiveTriangleInput[] = [];

    constructor(
        public readonly device: Device,
        meshes: SceneMeshDesc[],
        materials: SceneMaterialDesc[],
        lights: AnalyticLight[] = [],
        textureManager: TextureManager = new TextureManager(),
    ) {
        assert(meshes.length > 0 && materials.length > 0, "Scene requires geometry and materials");
        this.hasEmissiveMaterials = materials.some((m) => m.header?.emissive ?? false);

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
                globalMatrixID: meshID,
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

        // Node transforms: one node per mesh (globalMatrixID == mesh index).
        const world = new Float32Array(meshes.length * 16);
        const invT = new Float32Array(meshes.length * 16);
        meshes.forEach((mesh, i) => {
            const m = mesh.transform ?? float4x4.identity();
            world.set(m.toArray(), i * 16);
            invT.set(transpose(inverse(m)).toArray(), i * 16);
        });
        make("worldMatrices", world, 64);
        make("inverseTransposeWorldMatrices", invT, 64);

        // Software RT BVH over world-space triangles (DESIGN.md §5).
        const bvhTris: BvhTriangle[] = [];
        meshes.forEach((mesh, meshID) => {
            const m = mesh.transform ?? float4x4.identity();
            for (let p = 0; p < mesh.indices.length / 3; p++) {
                bvhTris.push({
                    v0: transformPoint(m, mesh.vertices[mesh.indices[p * 3]!]!.position),
                    v1: transformPoint(m, mesh.vertices[mesh.indices[p * 3 + 1]!]!.position),
                    v2: transformPoint(m, mesh.vertices[mesh.indices[p * 3 + 2]!]!.position),
                    instanceIndex: meshID,
                    primitiveIndex: p,
                });
            }
        });
        const bvh = buildBvh(bvhTris);
        make("bvhNodes", bvh.nodes, 16);
        make("bvhTris", bvh.tris, 16);

        // Analytic lights.
        this.lightCount = lights.length;
        make("lights", packLights(lights), 224);

        // Emissive geometry (LightCollection).
        const lc = buildLightCollection(
            meshes,
            materials.map((m) => {
                const em = m.basic.emissive;
                const factor = m.basic.emissiveFactor ?? 1;
                return {
                    emissive: m.header?.emissive ?? false,
                    radiance: [(em?.x ?? 0) * factor, (em?.y ?? 0) * factor, (em?.z ?? 0) * factor] as [number, number, number],
                };
            }),
        );
        this.emissiveTriangleCount = lc.triangleCount;
        this.emissiveMeshCount = lc.meshCount;
        this.emissiveFluxes = new Float32Array(lc.triangleCount);
        const fluxView = new DataView(lc.fluxData);
        for (let i = 0; i < lc.triangleCount; i++) this.emissiveFluxes[i] = fluxView.getFloat32(i * 32, true);
        // Retain builder inputs for the LightBVH sampler (native builds from the
        // UNPACKED emissive triangles: octahedral-decoded normals).
        const triView = new DataView(lc.triangleData);
        this.emissiveTriangles = Array.from({ length: lc.triangleCount }, (_v, i) => ({
            posW: [0, 1, 2].map((k) => [
                triView.getFloat32(i * 64 + k * 16, true),
                triView.getFloat32(i * 64 + k * 16 + 4, true),
                triView.getFloat32(i * 64 + k * 16 + 8, true),
            ]) as [Vec3, Vec3, Vec3],
            normal: decodeNormal2x16Host(triView.getUint32(i * 64 + 48, true)),
            flux: this.emissiveFluxes[i]!,
        }));
        make("emissiveTriangles", lc.triangleData, 64);
        make("emissiveFlux", lc.fluxData, 32);
        make("emissiveActiveTriangles", lc.activeTriangles, 4);
        make("emissiveTriToActive", lc.triToActiveMapping, 4);
        make("emissiveMeshData", lc.meshData, 16);
        make("emissivePerMeshInstanceOffset", lc.perMeshInstanceOffset, 4);

        // Materials.
        this.materialCount = materials.length;
        const blobBytes = new Uint8Array(materials.length * 128);
        materials.forEach((m, i) => {
            this.materialTypes.add(m.header?.materialType ?? MaterialType.Standard);
            blobBytes.set(packBasicMaterialBlob({ materialType: MaterialType.Standard, ...m.header }, m.basic), i * 128);
        });
        make("materialData", blobBytes, 128);
        make("materialBuffer0", new Uint32Array(4), 4);
        make("curveDummy", new Uint32Array(16), 32); // StaticCurveVertexData-sized dummy
        make("gridVolumeDummy", new Uint32Array(64), 256); // GridVolumeData-sized dummy (2x float4x4 + params)

        // Material textures packed into one array (DESIGN.md §6.2).
        const packed = textureManager.build(this.device);
        this.textureArray = packed.array;
        this.textureCount = Math.max(textureManager.count, 1);
        make("materialTextureUvScale", packed.uvScale, 8);
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

    setEnvMap(envMap: EnvMap | null): void {
        this.envMap = envMap;
    }

    /** Per-emissive-triangle flux in LightCollection order (for power sampling). */
    getEmissiveFluxes(): Float32Array {
        return this.emissiveFluxes;
    }

    /** Emissive triangles as LightBVH builder inputs (LightCollection order). */
    getEmissiveTriangles(): EmissiveTriangleInput[] {
        return this.emissiveTriangles;
    }

    getEnvMap(): EnvMap | null {
        return this.envMap;
    }

    /** Mirrors Scene::useAnalyticLights() (render settings default to enabled). */
    get useAnalyticLights(): boolean {
        return this.lightCount > 0;
    }

    /** Mirrors Scene::useEmissiveLights(). v1 checks material flags; the
     *  LightCollection active-triangle count refines this when NEE lands. */
    get useEmissiveLights(): boolean {
        return this.hasEmissiveMaterials;
    }

    /** Mirrors Scene::useEnvBackground(). */
    get useEnvBackground(): boolean {
        return this.envMap !== null;
    }

    /** Mirrors Scene::useEnvLight() (render settings default to enabled). */
    get useEnvLight(): boolean {
        return this.envMap !== null && this.envMap.intensity > 0;
    }

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
            MATERIAL_SYSTEM_TEXTURE_DESC_COUNT: this.textureCount,
            MATERIAL_SYSTEM_BUFFER_DESC_COUNT: 1,
            MATERIAL_SYSTEM_TEXTURE_3D_DESC_COUNT: 1,
            MATERIAL_SYSTEM_UDIM_INDIRECTION_ENABLED: 0,
            MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS: 0,
            MATERIAL_SYSTEM_USE_LIGHT_PROFILE: 0,
            FALCOR_MATERIAL_INSTANCE_SIZE: 256,
            // Static material dispatch (MaterialFactory override) — mirrors
            // MaterialSystem::getTypeConformances() type registration.
            WEBFALCOR_MTL_STANDARD: this.materialTypes.has(MaterialType.Standard) ? 1 : 0,
            WEBFALCOR_MTL_CLOTH: this.materialTypes.has(MaterialType.Cloth) ? 1 : 0,
            WEBFALCOR_MTL_HAIR: this.materialTypes.has(MaterialType.Hair) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIFFUSE: this.materialTypes.has(MaterialType.PBRTDiffuse) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIFFUSE_TRANSMISSION: this.materialTypes.has(MaterialType.PBRTDiffuseTransmission) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_CONDUCTOR: this.materialTypes.has(MaterialType.PBRTConductor) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_DIELECTRIC: this.materialTypes.has(MaterialType.PBRTDielectric) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_COATED_CONDUCTOR: this.materialTypes.has(MaterialType.PBRTCoatedConductor) ? 1 : 0,
            WEBFALCOR_MTL_PBRT_COATED_DIFFUSE: this.materialTypes.has(MaterialType.PBRTCoatedDiffuse) ? 1 : 0,
            SCENE_DIFFUSE_ALBEDO_MULTIPLIER: "1.0",
            FALCOR_NVAPI_AVAILABLE: 0,
            SAMPLE_GENERATOR_TYPE: 0, // TinyUniform (SampleGeneratorType.slangh)
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
        c["viewProjMatNoJitter"] = cam.viewProjMatNoJitter;
        c["prevViewProjMatNoJitter"] = cam.prevViewProjMatNoJitter;
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
        scene["webfalcorBvhNodes"] = this.buffers["bvhNodes"]!;
        scene["webfalcorBvhTris"] = this.buffers["bvhTris"]!;
        scene["lights"] = this.buffers["lights"]!;
        scene["lightCount"] = this.lightCount;
        scene["prevVertices"] = this.buffers["vertices"]!;
        // Curve buffers (no curve geometry yet; dummies for DCE survivors).
        scene["curveVertices"] = this.buffers["curveDummy"]!;
        scene["prevCurveVertices"] = this.buffers["curveDummy"]!;
        scene["curveIndices"] = this.buffers["curveDummy"]!;
        try {
            // Only referenced by volume-aware passes (binding absent otherwise).
            scene["gridVolumes"] = this.buffers["gridVolumeDummy"]!;
        } catch {
            /* binding absent in this variant */
        }
        scene["indexData"]["data0"] = this.buffers["indices"]!;

        // Env map (dummy black texture + zeroed uniforms when absent).
        if (this.envMap) {
            this.envMap.bindShaderData(scene["envMap"] as ShaderVar);
        } else {
            scene["envMap"]["envMap"] = this.dummyTexture;
            scene["envMap"]["envSampler"] = this.sampler;
        }

        // Emissive geometry (LightCollection.slang).
        const lightCollection = scene["lightCollection"];
        lightCollection["triangleCount"] = this.emissiveTriangleCount;
        lightCollection["activeTriangleCount"] = this.emissiveTriangleCount;
        lightCollection["meshCount"] = this.emissiveMeshCount;
        lightCollection["triangleData"] = this.buffers["emissiveTriangles"]!;
        lightCollection["activeTriangles"] = this.buffers["emissiveActiveTriangles"]!;
        lightCollection["triToActiveMapping"] = this.buffers["emissiveTriToActive"]!;
        lightCollection["fluxData"] = this.buffers["emissiveFlux"]!;
        lightCollection["meshData"] = this.buffers["emissiveMeshData"]!;
        lightCollection["perMeshInstanceOffset"] = this.buffers["emissivePerMeshInstanceOffset"]!;

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
