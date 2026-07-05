/**
 * LightCollection GPU-data builder mirroring Falcor/Scene/Lights/LightCollection.cpp
 * (BuildTriangleList + FinalizeIntegration) for untextured emissive basic
 * materials: averageRadiance = emissive * emissiveFactor and
 * flux = luminance(averageRadiance) * area * pi. Emissive-texture integration
 * comes with the texture-LOD work.
 *
 * Native runs this on the GPU; our scenes are static so a CPU build at scene
 * creation produces identical data (world-space positions, face normal
 * convention and encodings match Scene.slang computeFaceNormalAndAreaW and
 * PackedEmissiveTriangle.pack).
 */

import { float3 } from "../../Utils/Math/Vector.js";
import { float4x4, transformPoint } from "../../Utils/Math/Matrix.js";
import { encodeNormal2x16 } from "../SceneData.js";
import { f32tof16 } from "../Material/MaterialData.js";
import type { SceneMeshDesc } from "../Scene.js";

export const kInvalidIndex = 0xffffffff;

export interface EmissiveMaterialInfo {
    emissive: boolean;
    radiance: [number, number, number]; // emissive color * emissiveFactor
}

export interface LightCollectionData {
    triangleCount: number;
    meshCount: number;
    /** PackedEmissiveTriangle[], 64B stride. */
    triangleData: ArrayBuffer;
    /** EmissiveFlux[], 32B stride (WGSL vec3 alignment: flux@0, averageRadiance@16). */
    fluxData: ArrayBuffer;
    activeTriangles: Uint32Array;
    triToActiveMapping: Uint32Array;
    /** MeshLightData[], 4 uints each. */
    meshData: Uint32Array;
    perMeshInstanceOffset: Uint32Array;
}

export function buildLightCollection(meshes: SceneMeshDesc[], materials: EmissiveMaterialInfo[]): LightCollectionData {
    interface Tri {
        posW: float3[];
        uv: [number, number][];
        normal: float3;
        area: number;
        materialID: number;
        lightIdx: number;
    }
    const tris: Tri[] = [];
    const meshLights: number[] = []; // instanceID, triangleOffset, triangleCount, materialID
    const perMeshInstanceOffset = new Uint32Array(meshes.length).fill(kInvalidIndex);

    meshes.forEach((mesh, instanceID) => {
        const mat = materials[mesh.materialID];
        if (!mat?.emissive) return;
        const lightIdx = meshLights.length / 4;
        const triangleOffset = tris.length;
        perMeshInstanceOffset[instanceID] = triangleOffset;
        const world = mesh.transform ?? float4x4.identity();
        for (let t = 0; t + 2 < mesh.indices.length; t += 3) {
            const p = [0, 1, 2].map((k) => transformPoint(world, mesh.vertices[mesh.indices[t + k]!]!.position));
            const uv = [0, 1, 2].map((k) => {
                const c = mesh.vertices[mesh.indices[t + k]!]!.texCrd;
                return [c.x, c.y] as [number, number];
            });
            // computeFaceNormalAndAreaW: N = cross(p1-p0, p2-p0), area = |N|/2.
            const e0 = new float3(p[1]!.x - p[0]!.x, p[1]!.y - p[0]!.y, p[1]!.z - p[0]!.z);
            const e1 = new float3(p[2]!.x - p[0]!.x, p[2]!.y - p[0]!.y, p[2]!.z - p[0]!.z);
            const n = new float3(e0.y * e1.z - e0.z * e1.y, e0.z * e1.x - e0.x * e1.z, e0.x * e1.y - e0.y * e1.x);
            const len = Math.hypot(n.x, n.y, n.z);
            const area = 0.5 * len;
            const normal = len > 0 ? new float3(n.x / len, n.y / len, n.z / len) : new float3(0, 0, 1);
            tris.push({ posW: p, uv, normal, area, materialID: mesh.materialID, lightIdx });
        }
        meshLights.push(instanceID, triangleOffset, tris.length - triangleOffset, mesh.materialID);
    });

    const triangleData = new ArrayBuffer(Math.max(tris.length, 1) * 64);
    const fluxData = new ArrayBuffer(Math.max(tris.length, 1) * 32);
    const tv = new DataView(triangleData);
    const fv = new DataView(fluxData);
    tris.forEach((tri, i) => {
        const base = i * 64;
        for (let k = 0; k < 3; k++) {
            tv.setFloat32(base + k * 16, tri.posW[k]!.x, true);
            tv.setFloat32(base + k * 16 + 4, tri.posW[k]!.y, true);
            tv.setFloat32(base + k * 16 + 8, tri.posW[k]!.z, true);
            const enc = ((f32tof16(tri.uv[k]![1]) << 16) | f32tof16(tri.uv[k]![0])) >>> 0;
            tv.setUint32(base + k * 16 + 12, enc, true);
        }
        tv.setUint32(base + 48, encodeNormal2x16(tri.normal) >>> 0, true);
        tv.setFloat32(base + 52, tri.area, true);
        tv.setUint32(base + 56, tri.materialID, true);
        tv.setUint32(base + 60, tri.lightIdx, true);

        const rad = materials[tri.materialID]!.radiance;
        const flux = (0.2126 * rad[0] + 0.7152 * rad[1] + 0.0722 * rad[2]) * tri.area * Math.PI;
        fv.setFloat32(i * 32, flux, true);
        fv.setFloat32(i * 32 + 16, rad[0], true);
        fv.setFloat32(i * 32 + 20, rad[1], true);
        fv.setFloat32(i * 32 + 24, rad[2], true);
    });

    // All triangles are active (native culls zero-flux triangles; radiance is
    // uniform per material here, so emissive materials never yield zero flux).
    const active = new Uint32Array(Math.max(tris.length, 1));
    const mapping = new Uint32Array(Math.max(tris.length, 1));
    for (let i = 0; i < tris.length; i++) {
        active[i] = i;
        mapping[i] = i;
    }

    return {
        triangleCount: tris.length,
        meshCount: meshLights.length / 4,
        triangleData,
        fluxData,
        activeTriangles: active,
        triToActiveMapping: mapping,
        meshData: meshLights.length > 0 ? new Uint32Array(meshLights) : new Uint32Array([kInvalidIndex, kInvalidIndex, 0, kInvalidIndex]),
        perMeshInstanceOffset: perMeshInstanceOffset.length > 0 ? perMeshInstanceOffset : new Uint32Array([kInvalidIndex]),
    };
}
