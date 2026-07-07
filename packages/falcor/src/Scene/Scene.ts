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
import { Sampler, TextureFilteringMode, TextureAddressingMode } from "../Core/API/Sampler.js";
import { ResourceBindFlags, MemoryType, ResourceType } from "../Core/API/Types.js";
import { ResourceFormat } from "../Core/API/Formats.js";
import { DefineList } from "../Core/Program/DefineList.js";
import type { ShaderVar } from "../Core/Program/ParameterBlock.js";
import { Camera } from "./Camera/Camera.js";
import { float4x4, transpose, inverse } from "../Utils/Math/Matrix.js";
import { buildBvh, buildAabbBvh, type BvhTriangle } from "./SoftwareRT/Bvh.js";
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
import { assert, RuntimeError } from "../Core/Error.js";
import type { NDSDFGrid } from "./SDFs/NDSDFGrid.js";
import { SDFSBS } from "./SDFs/SDFSBS.js";
import { SDFSVS } from "./SDFs/SDFSVS.js";

/** One SDF grid instance (mirrors Scene::mSDFGrids + mSDFGridDesc + instance). */
export interface SceneSDFGridDesc {
    grid: NDSDFGrid | SDFSBS | SDFSVS;
    materialID: number;
    transform?: float4x4;
}

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
    readonly gridVolumes: import("./Volume/GridVolume.js").GridVolume[] = [];
    private gridCount = 0;
    private grid0Stats: { minIndex: [number, number, number]; minValue: number; maxIndex: [number, number, number]; maxValue: number } | null = null;
    private buffers: Record<string, Buffer> = {};
    private textureArray: Texture;
    private textureArrayLinear: Texture;
    private dummyTexture: Texture;
    private texture3D: Texture;
    private sampler: Sampler;
    private materialCount = 0;
    private instanceCount = 0;
    private textureCount = 1;
    private drawList: { indexCount: number; firstIndex: number; baseVertex: number; firstInstance: number }[] = [];

    private lightCount = 0;
    /** Analytic light descriptors (RTXDI needs types/order). */
    readonly analyticLights: AnalyticLight[] = [];
    emissiveActiveTriangleCount = 0;
    private bvhTrisOffset = 0;
    private invTransposeOffset = 0;
    readonly sdfGrids: SceneSDFGridDesc[];
    private sdfInstanceFirst = 0;
    private sdfAtlasTexture: Texture | null = null;
    private sdfSampler: Sampler | null = null;
    private sbsResources: { aabbs: Buffer; indirection: Texture; bricks: Texture; sampler: Sampler } | null = null;
    private svsResources: { voxels: Buffer } | null = null;
    private sdfBvhBuffers: { buf: Buffer; primOffset: number } | null = null;
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
        sdfGrids: SceneSDFGridDesc[] = [],
    ) {
        this.sdfGrids = sdfGrids;
        // Geometry-less scenes are legal (pure-volume scenes like smoke.pyscene):
        // buffers pad to one zeroed struct and ray queries simply miss.
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
        // SDF grid instances append after the triangle instances (they are
        // not in the triangle BVH; SBS/SVS use a separate primitive-AABB BVH).
        this.sdfInstanceFirst = instances.length;
        sdfGrids.forEach((desc, i) => {
            instances.push({
                type: GeometryType.SDFGrid,
                globalMatrixID: meshes.length + i,
                materialID: desc.materialID,
                geometryID: i,
                vbOffset: 0,
                ibOffset: 0,
                instanceIndex: meshes.length + i,
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
            let bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
            // WebGPU forbids zero-size storage bindings: pad empty buffers to
            // one zeroed struct (counts gate all shader-side access).
            if (bytes.byteLength === 0) bytes = new Uint8Array(Math.max(structSize, 4));
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

        // Node transforms: one node per mesh, then one per SDF grid instance
        // (globalMatrixID indexes this order). One merged buffer
        // (16-storage-buffer budget, same pattern as the BVH merge): world
        // matrices then inverse-transpose matrices.
        const nodeCount = meshes.length + sdfGrids.length;
        const world = new Float32Array(nodeCount * 2 * 16);
        const putNode = (i: number, m: float4x4) => {
            world.set(m.toArray(), i * 16);
            world.set(transpose(inverse(m)).toArray(), (nodeCount + i) * 16);
        };
        meshes.forEach((mesh, i) => putNode(i, mesh.transform ?? float4x4.identity()));
        sdfGrids.forEach((desc, i) => putNode(meshes.length + i, desc.transform ?? float4x4.identity()));
        make("worldMatrices", world, 64);
        this.invTransposeOffset = nodeCount;

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
        // One merged buffer (16-storage-buffer budget): nodes then triangles.
        const bvhMerged = new Float32Array(bvh.nodes.length + bvh.tris.length);
        bvhMerged.set(bvh.nodes, 0);
        bvhMerged.set(bvh.tris, bvh.nodes.length);
        this.bvhTrisOffset = bvh.nodes.length / 4;
        make("bvhNodes", bvhMerged, 16);

        // Analytic lights.
        this.lightCount = lights.length;
        this.analyticLights = lights;
        make("lights", packLights(lights), 224);

        // Emissive geometry (LightCollection).
        const lc = buildLightCollection(
            meshes,
            materials.map((m) => {
                const em = m.basic.emissive;
                const factor = m.basic.emissiveFactor ?? 1;
                // Textured emissives integrate per triangle (EmissiveIntegrator
                // semantics); texture handle low bits = textureID (mode bits high).
                const texEmissive = m.basic.texEmissive;
                const emissiveTexture =
                    texEmissive !== undefined ? (textureManager.readLinearPixels(texEmissive & 0x1fffffff) ?? undefined) : undefined;
                return {
                    emissive: m.header?.emissive ?? false,
                    radiance: [(em?.x ?? 0) * factor, (em?.y ?? 0) * factor, (em?.z ?? 0) * factor] as [number, number, number],
                    emissiveTexture,
                    emissiveFactor: factor,
                };
            }),
        );
        this.emissiveTriangleCount = lc.triangleCount;
        this.emissiveActiveTriangleCount = lc.activeTriangles.length;
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
        this.textureArrayLinear = packed.arrayLinear;
        this.textureCount = Math.max(textureManager.count, 1);
        make("materialTextureUvScale", packed.texInfo, 16);
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
    /** Binds gScene.sdfGrid0 for the NormalizedDenseGrid implementation. */
    private bindSdfNd(scene: ShaderVar, grid: NDSDFGrid): void {
        if (!this.sdfAtlasTexture) {
            const atlas = grid.buildAtlas();
            this.sdfAtlasTexture = new Texture(this.device, {
                type: ResourceType.Texture3D,
                width: atlas.width,
                height: atlas.height,
                depth: atlas.depth,
                format: ResourceFormat.R8Snorm,
                bindFlags: ResourceBindFlags.ShaderResource,
                name: "Scene::sdfGrid0Atlas",
            });
            this.sdfAtlasTexture.setSubresourceBlob(0, 0, new Uint8Array(atlas.data.buffer));
            // Native NDSDFGrid::SharedData sampler: linear min/mag/mip, clamp.
            this.sdfSampler = new Sampler(this.device, {
                magFilter: TextureFilteringMode.Linear,
                minFilter: TextureFilteringMode.Linear,
                mipFilter: TextureFilteringMode.Linear,
                addressModeU: TextureAddressingMode.Clamp,
                addressModeV: TextureAddressingMode.Clamp,
                addressModeW: TextureAddressingMode.Clamp,
            });
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["atlasTexture"] = this.sdfAtlasTexture;
            v["sampler"] = this.sdfSampler!;
            v["lodCount"] = grid.lodCount;
            v["coarsestLODAsLevel"] = grid.coarsestLODAsLevel;
            v["coarsestLODGridWidth"] = grid.coarsestLODGridWidth;
            v["coarsestLODNormalizationFactor"] = grid.coarsestLODNormalizationFactor;
            v["narrowBandThickness"] = grid.narrowBandThickness;
        } catch (e) {
            console.error(`# sdfGrid0 (ND) bind failed: ${e}`);
        }
    }

    /** Binds gScene.sdfGrid0 for the SparseBrickSet implementation. */
    private bindSdfSbs(scene: ShaderVar, grid: SDFSBS): void {
        if (!this.sbsResources) {
            const storage = ResourceBindFlags.ShaderResource;
            // AABB StructuredBuffer: 32-byte stride (min.xyz @0, max.xyz @16).
            const aabbData = new Float32Array(grid.aabbs.length * 8);
            grid.aabbs.forEach((a, i) => {
                aabbData.set(a.min, i * 8);
                aabbData.set(a.max, i * 8 + 4);
            });
            const aabbs = new Buffer(this.device, {
                size: Math.max(aabbData.byteLength, 32),
                structSize: 32,
                bindFlags: storage | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal,
                name: "Scene::sdfGrid0Aabbs",
            });
            aabbs.setBlob(new Uint8Array(aabbData.buffer));

            const vbpa = grid.virtualBricksPerAxis;
            const indirection = new Texture(this.device, {
                type: ResourceType.Texture3D,
                width: vbpa,
                height: vbpa,
                depth: vbpa,
                format: ResourceFormat.R32Uint,
                bindFlags: storage,
                name: "Scene::sdfGrid0Indirection",
            });
            indirection.setSubresourceBlob(0, 0, new Uint8Array(grid.indirection.buffer));

            const bricks = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: grid.brickTextureDimensions[0],
                height: grid.brickTextureDimensions[1],
                format: ResourceFormat.R32Float,
                bindFlags: storage,
                name: "Scene::sdfGrid0Bricks",
            });
            bricks.setSubresourceBlob(0, 0, new Uint8Array(grid.brickTexture.buffer));

            // Native SDFSBS::SharedData sampler: linear, clamp (brick edges).
            const sampler = new Sampler(this.device, {
                magFilter: TextureFilteringMode.Linear,
                minFilter: TextureFilteringMode.Linear,
                mipFilter: TextureFilteringMode.Linear,
                addressModeU: TextureAddressingMode.Clamp,
                addressModeV: TextureAddressingMode.Clamp,
                addressModeW: TextureAddressingMode.Clamp,
            });
            this.sbsResources = { aabbs, indirection, bricks, sampler };
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["aabbs"] = this.sbsResources.aabbs;
            v["indirectionBuffer"] = this.sbsResources.indirection;
            v["bricks"] = this.sbsResources.bricks;
            v["sampler"] = this.sbsResources.sampler;
            v["virtualGridWidth"] = grid.gridWidth;
            v["virtualBricksPerAxis"] = grid.virtualBricksPerAxis;
            v["bricksPerAxis"] = grid.bricksPerAxis;
            v["brickTextureDimensions"] = grid.brickTextureDimensions;
            v["brickWidth"] = grid.brickWidth;
            v["normalizationFactor"] = grid.normalizationFactor;
        } catch (e) {
            console.error(`# sdfGrid0 (SBS) bind failed: ${e}`);
        }
    }

    /** Binds gScene.sdfGrid0 for the SparseVoxelSet implementation. */
    private bindSdfSvs(scene: ShaderVar, grid: SDFSVS): void {
        if (!this.svsResources) {
            const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
            // WebFalcorSVSVoxel StructuredBuffer: AABB (32B) + SDFSVSVoxel (80B)
            // merged, 112-byte stride (the SVS.slang override reads .aabb/.voxel).
            const n = Math.max(grid.voxelCount, 1);
            const combined = new Float32Array(n * 28); // 112 bytes = 28 words
            const combinedU = new Uint32Array(combined.buffer);
            for (let i = 0; i < grid.voxelCount; i++) {
                const a = grid.aabbs[i]!;
                combined.set(a.min, i * 28); // aabb.min @0
                combined.set(a.max, i * 28 + 4); // aabb.max @16
                combinedU.set(grid.voxelData.subarray(i * 20, i * 20 + 20), i * 28 + 8); // voxel @32
            }
            const voxels = new Buffer(this.device, {
                size: combined.byteLength,
                structSize: 112,
                bindFlags: storage,
                memoryType: MemoryType.DeviceLocal,
                name: "Scene::sdfGrid0Voxels",
            });
            voxels.setBlob(new Uint8Array(combined.buffer));
            this.svsResources = { voxels };
        }
        try {
            const v = scene["sdfGrid0"] as ShaderVar;
            v["voxels"] = this.svsResources.voxels;
            v["virtualGridWidth"] = grid.gridWidth;
            v["oneDivVirtualGridWidth"] = 1 / grid.gridWidth;
            v["normalizationFactor"] = grid.normalizationFactor;
        } catch (e) {
            console.error(`# sdfGrid0 (SVS) bind failed: ${e}`);
        }
    }

    getSceneDefines(): DefineList {
        return new DefineList().addAll({
            SCENE_GEOMETRY_TYPES: (1 << GeometryType.TriangleMesh) | (this.sdfGrids.length > 0 ? 1 << GeometryType.SDFGrid : 0),
            SCENE_GRID_COUNT: this.gridCount,
            // Mirrors Scene::getSceneSDFGridDefines (defaults for all types:
            // VoxelSphereTracing, NumericDiscontinuous, 256 iterations).
            SCENE_SDF_GRID_COUNT: this.sdfGrids.length,
            SCENE_SDF_GRID_MAX_LOD_COUNT: this.sdfGrids.length > 0 ? Math.max(...this.sdfGrids.map((g) => (g.grid instanceof SDFSBS || g.grid instanceof SDFSVS ? 32 - Math.clz32(g.grid.gridWidth) : g.grid.lodCount))) : 0,
            // 1 = NormalizedDenseGrid, 2 = SparseVoxelSet, 3 = SparseBrickSet (all grids in a scene share a type).
            SCENE_SDF_GRID_IMPLEMENTATION: this.sdfGrids.length > 0 ? (this.sdfGrids[0]!.grid instanceof SDFSBS ? 3 : this.sdfGrids[0]!.grid instanceof SDFSVS ? 2 : 1) : 0,
            SCENE_SDF_GRID_IMPLEMENTATION_NDSDF: 1,
            SCENE_SDF_GRID_IMPLEMENTATION_SVS: 2,
            SCENE_SDF_GRID_IMPLEMENTATION_SBS: 3,
            SCENE_SDF_GRID_IMPLEMENTATION_SVO: 4,
            SCENE_SDF_NO_INTERSECTION_METHOD: 0,
            SCENE_SDF_NO_VOXEL_SOLVER: 1,
            SCENE_SDF_VOXEL_SPHERE_TRACING: 2,
            SCENE_SDF_NO_GRADIENT_EVALUATION_METHOD: 0,
            SCENE_SDF_GRADIENT_NUMERIC_DISCONTINUOUS: 1,
            SCENE_SDF_GRADIENT_NUMERIC_CONTINUOUS: 2,
            SCENE_SDF_VOXEL_INTERSECTION_METHOD: 2,
            SCENE_SDF_GRADIENT_EVALUATION_METHOD: 1,
            SCENE_SDF_SOLVER_MAX_ITERATION_COUNT: 256,
            SCENE_SDF_OPTIMIZE_VISIBILITY_RAYS: 1,
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
            // Material-less scenes (pure volumes) still need one registered type:
            // the factory's fallback return must exist for WGSL (E41009).
            WEBFALCOR_MTL_STANDARD: this.materialTypes.has(MaterialType.Standard) || this.materialTypes.size === 0 ? 1 : 0,
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

    /**
     * Uploads grid-volume GPU data after resolve() populates gridVolumes
     * (web divergence: volumes load asynchronously after construction).
     * One grid supported (gScene.grid0 — WGSL has no binding arrays).
     */
    finalizeGridVolumes(): void {
        if (this.gridVolumes.length === 0) return;
        const grids: import("./Volume/Grid.js").Grid[] = [];
        const gridIndex = (g: import("./Volume/Grid.js").Grid | undefined): number => {
            if (!g) return 0xffffffff;
            let i = grids.indexOf(g);
            if (i < 0) {
                i = grids.length;
                grids.push(g);
            }
            return i;
        };

        // GridVolumeData: 192 B per volume (2x float4x4 + 4x 16 B rows).
        const data = new ArrayBuffer(this.gridVolumes.length * 192);
        const f32 = new Float32Array(data);
        const u32 = new Uint32Array(data);
        this.gridVolumes.forEach((vol, vi) => {
            const o = vi * 48; // floats
            // Mirrors Scene::updateGridVolumes: the uploaded transform merges the
            // volume transform (identity: no animation yet) with the density
            // grid's index->world map, so invTransform maps world -> INDEX space.
            const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
            f32.set(vol.densityGrid ? vol.densityGrid.indexToWorldMatrix : identity, o);
            f32.set(vol.densityGrid ? vol.densityGrid.worldToIndexMatrix : identity, o + 16);
            const b = vol.bounds ?? { min: [0, 0, 0], max: [0, 0, 0] };
            f32.set(b.min, o + 32);
            f32[o + 35] = vol.densityScale;
            f32.set(b.max, o + 36);
            f32[o + 39] = vol.emissionScale;
            u32[o + 40] = gridIndex(vol.densityGrid);
            u32[o + 41] = gridIndex(vol.emissionGrid);
            u32[o + 42] = 0; // flags (emission mode Direct)
            f32[o + 43] = vol.anisotropy;
            f32[o + 44] = vol.albedo.x;
            f32[o + 45] = vol.albedo.y;
            f32[o + 46] = vol.albedo.z;
            f32[o + 47] = vol.emissionTemperature;
        });
        const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        const volBuf = new Buffer(this.device, { size: data.byteLength, structSize: 192, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::gridVolumes" });
        volBuf.setBlob(new Uint8Array(data));
        this.buffers["gridVolumesData"] = volBuf;

        if (grids.length > 1) throw new RuntimeError("Scene: only one grid supported (gScene.grid0; WGSL has no binding arrays)");
        if (grids.length === 1) {
            const g = grids[0]!;
            const buf = new Buffer(this.device, { size: g.gridBuffer.byteLength, structSize: 4, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::grid0" });
            buf.setBlob(g.gridBuffer);
            this.buffers["grid0"] = buf;
            this.grid0Stats = { minIndex: g.minIndex, minValue: g.minValue, maxIndex: g.maxIndex, maxValue: g.maxValue };
        }
        this.gridCount = grids.length;
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
        scene["prevWorldMatrices"] = this.buffers["worldMatrices"]!;
        scene["webfalcorInvTransposeOffset"] = this.invTransposeOffset;
        scene["geometryInstances"] = this.buffers["geometryInstances"]!;
        scene["meshes"] = this.buffers["meshes"]!;
        scene["vertices"]["data0"] = this.buffers["vertices"]!;
        scene["webfalcorBvhNodes"] = this.buffers["bvhNodes"]!;
        scene["webfalcorBvhTrisOffset"] = this.bvhTrisOffset;
        scene["lights"] = this.buffers["lights"]!;
        scene["lightCount"] = this.lightCount;
        scene["prevVertices"] = this.buffers["vertices"]!;
        // Curve buffers (no curve geometry yet; dummies for DCE survivors).
        scene["curveVertices"] = this.buffers["curveDummy"]!;
        scene["prevCurveVertices"] = this.buffers["curveDummy"]!;
        scene["curveIndices"] = this.buffers["curveDummy"]!;
        try {
            // Only referenced by volume-aware passes (binding absent otherwise).
            scene["gridVolumeCount"] = this.gridVolumes.length;
            scene["gridVolumes"] = this.buffers["gridVolumesData"] ?? this.buffers["gridVolumeDummy"]!;
        } catch (e) {
            if (this.gridVolumes.length > 0) console.error(`# gridVolumes bind failed: ${e}`);
        }
        scene["indexData"]["data0"] = this.buffers["indices"]!;

        // SDF grids (one instance set; sdfGrid0 bindings survive only with
        // SCENE_SDF_GRID_COUNT > 0 — trySet semantics via try/catch).
        const grid0 = this.sdfGrids.length > 0 ? this.sdfGrids[0]!.grid : null;
        const sbsGrid = grid0 instanceof SDFSBS ? grid0 : null;
        const svsGrid = grid0 instanceof SDFSVS ? grid0 : null;
        // SBS/SVS traverse a BVH over their primitive AABBs (bricks/voxels).
        const sdfAabbs = sbsGrid ? sbsGrid.aabbs : svsGrid ? svsGrid.aabbs : null;
        try {
            scene["webfalcorSdfInstanceFirst"] = this.sdfInstanceFirst;
            scene["webfalcorSdfInstanceCount"] = this.sdfGrids.length;
            scene["webfalcorSdfBrickCount"] = sdfAabbs ? sdfAabbs.length : 0;
        } catch {
            /* SDF-less kernel variant */
        }
        if (sdfAabbs) {
            if (!this.sdfBvhBuffers) {
                const bvh = buildAabbBvh(sdfAabbs);
                // One merged float4 buffer (16-storage-buffer budget): BVH nodes
                // (2 float4/node), then prim indices packed 4-per-float4.
                const nodeFloat4s = bvh.nodes.length / 4;
                const primFloat4s = Math.ceil(bvh.primIndices.length / 4);
                const merged = new Float32Array((nodeFloat4s + primFloat4s) * 4);
                merged.set(bvh.nodes, 0);
                new Uint32Array(merged.buffer).set(bvh.primIndices, nodeFloat4s * 4);
                const storage = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
                const buf = new Buffer(this.device, { size: merged.byteLength, structSize: 16, bindFlags: storage, memoryType: MemoryType.DeviceLocal, name: "Scene::sdfBvh" });
                buf.setBlob(new Uint8Array(merged.buffer));
                this.sdfBvhBuffers = { buf, primOffset: nodeFloat4s };
            }
            try {
                scene["webfalcorSdfBvh"] = this.sdfBvhBuffers.buf;
                scene["webfalcorSdfPrimOffset"] = this.sdfBvhBuffers.primOffset;
            } catch {
                /* bindings absent in NDSDF/SVO kernel variant */
            }
        }
        if (this.sdfGrids.length > 0) {
            if (this.sdfGrids.length > 1) throw new RuntimeError("Scene: only one SDF grid supported (gScene.sdfGrid0; WGSL has no binding arrays)");
            if (sbsGrid) this.bindSdfSbs(scene, sbsGrid);
            else if (svsGrid) this.bindSdfSvs(scene, svsGrid);
            else this.bindSdfNd(scene, this.sdfGrids[0]!.grid as NDSDFGrid);
        }

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

        // Grid volume single instance (NanoVDB buffer when loaded; dummies keep
        // SCENE_GRID_COUNT=0 variants bindable). Bricked-grid textures stay
        // dummies: the upstream consumers use the NanoVDB lookup path.
        scene["grid0"]["buf"] = this.buffers["grid0"] ?? this.buffers["materialBuffer0"]!;
        scene["grid0"]["rangeTex"] = this.gridRangeTex;
        scene["grid0"]["indirectionTex"] = this.gridIndirectionTex;
        scene["grid0"]["atlasTex"] = this.gridAtlasTex;
        if (this.grid0Stats) {
            scene["grid0"]["minIndex"] = this.grid0Stats.minIndex;
            scene["grid0"]["minValue"] = this.grid0Stats.minValue;
            scene["grid0"]["maxIndex"] = this.grid0Stats.maxIndex;
            scene["grid0"]["maxValue"] = this.grid0Stats.maxValue;
        }

        // Light profile (disabled; dummy bindings).
        scene["materials"]["lightProfile"]["texture"] = this.dummyTexture;
        scene["materials"]["lightProfile"]["sampler"] = this.sampler;

        // Material system.
        const materials = scene["materials"];
        materials["materialCount"] = this.materialCount;
        materials["materialData"] = this.buffers["materialData"]!;
        materials["materialSampler0"] = this.sampler;
        materials["materialTexturesArray"] = this.textureArray;
        materials["materialTexturesArrayLinear"] = this.textureArrayLinear;
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
