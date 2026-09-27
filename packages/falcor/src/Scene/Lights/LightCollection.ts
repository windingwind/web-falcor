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
    /** Emissive texture (linear RGB texels) for per-triangle integration. */
    emissiveTexture?: { width: number; height: number; rgb: Float32Array };
    emissiveFactor?: number;
}

/**
 * Mirrors EmissiveIntegrator.3d.slang + FinalizeIntegration.cs.slang: the
 * triangle is rasterized in texture space with one sample per texel and
 * ANALYTIC edge coverage (triangle-vs-texel-square clipped polygon area);
 * averageEmissive = sum(w * texel) / sum(w). Degenerate UV triangles fall
 * back to the average of the three vertex point samples.
 */
function integrateEmissiveTexture(
    tex: { width: number; height: number; rgb: Float32Array },
    uv: [number, number][],
): [number, number, number] {
    const w = tex.width;
    const h = tex.height;
    // Texel-space vertices, offset to positive (uvOffset = floor(uvMin)).
    const uMin = Math.min(uv[0]![0], uv[1]![0], uv[2]![0]);
    const vMin = Math.min(uv[0]![1], uv[1]![1], uv[2]![1]);
    const off = [Math.floor(uMin), Math.floor(vMin)];
    const px = uv.map((c) => [(c[0] - off[0]!) * w, (c[1] - off[1]!) * h]);

    // Sutherland-Hodgman against the texel square, ping-ponging two fixed buffers (a triangle clips to <= 7 vertices).
    const bufA = new Float64Array(16);
    const bufB = new Float64Array(16);
    // side: 0 x >= c, 1 x <= c, 2 y >= c, 3 y <= c; returns the output vertex count.
    const clipEdge = (src: Float64Array, n: number, dst: Float64Array, side: number, c: number): number => {
        const axis = side >> 1;
        let m = 0;
        for (let i = 0; i < n; i++) {
            const j = i + 1 === n ? 0 : i + 1;
            const ax = src[i * 2]!, ay = src[i * 2 + 1]!, bx = src[j * 2]!, by = src[j * 2 + 1]!;
            const av = axis === 0 ? ax : ay;
            const bv = axis === 0 ? bx : by;
            const ain = side & 1 ? av <= c : av >= c;
            const bin = side & 1 ? bv <= c : bv >= c;
            if (ain) {
                dst[m * 2] = ax;
                dst[m * 2 + 1] = ay;
                m++;
            }
            if (ain !== bin) {
                const t = axis === 0 ? (c - ax) / (bx - ax) : (c - ay) / (by - ay);
                dst[m * 2] = ax + (bx - ax) * t;
                dst[m * 2 + 1] = ay + (by - ay) * t;
                m++;
            }
        }
        return m;
    };
    const clipArea = (tx: number, ty: number): number => {
        for (let i = 0; i < 3; i++) [bufA[i * 2], bufA[i * 2 + 1]] = [px[i]![0]!, px[i]![1]!];
        let n = clipEdge(bufA, 3, bufB, 0, tx);
        n = clipEdge(bufB, n, bufA, 1, tx + 1);
        n = clipEdge(bufA, n, bufB, 2, ty);
        n = clipEdge(bufB, n, bufA, 3, ty + 1);
        let area = 0;
        for (let i = 0; i < n; i++) {
            const j = i + 1 === n ? 0 : i + 1;
            area += bufA[i * 2]! * bufA[j * 2 + 1]! - bufA[j * 2]! * bufA[i * 2 + 1]!;
        }
        return Math.abs(area) * 0.5;
    };

    const texel = (ix: number, iy: number): [number, number, number] => {
        // Wrap addressing (native samples with a wrap point sampler).
        const x = ((ix % w) + w) % w;
        const y = ((iy % h) + h) % h;
        const i = (y * w + x) * 3;
        return [tex.rgb[i]!, tex.rgb[i + 1]!, tex.rgb[i + 2]!];
    };

    const x0 = Math.floor(Math.min(px[0]![0]!, px[1]![0]!, px[2]![0]!));
    const x1 = Math.ceil(Math.max(px[0]![0]!, px[1]![0]!, px[2]![0]!));
    const y0 = Math.floor(Math.min(px[0]![1]!, px[1]![1]!, px[2]![1]!));
    const y1 = Math.ceil(Math.max(px[0]![1]!, px[1]![1]!, px[2]![1]!));

    let sr = 0;
    let sg = 0;
    let sb = 0;
    let sw = 0;
    for (let ty = y0; ty < y1; ty++) {
        for (let tx = x0; tx < x1; tx++) {
            const a = clipArea(tx, ty);
            if (a <= 0) continue;
            // Wrap addressing, as texel().
            const i = ((((ty % h) + h) % h) * w + (((tx % w) + w) % w)) * 3;
            sr += a * tex.rgb[i]!;
            sg += a * tex.rgb[i + 1]!;
            sb += a * tex.rgb[i + 2]!;
            sw += a;
        }
    }
    if (sw > 0) return [sr / sw, sg / sw, sb / sw];
    // Degenerate in texture space: average the three vertex samples.
    let r = 0;
    let g = 0;
    let b = 0;
    for (const c of uv) {
        const t = texel(Math.floor(c[0] * w), Math.floor(c[1] * h));
        r += t[0];
        g += t[1];
        b += t[2];
    }
    return [r / 3, g / 3, b / 3];
}

export interface LightCollectionData {
    triangleCount: number;
    meshCount: number;
    /** PackedEmissiveTriangle[], 64B stride. */
    triangleData: ArrayBuffer;
    /** EmissiveFlux[], 32B stride (WGSL vec3 alignment: flux@0, averageRadiance@16). */
    fluxData: ArrayBuffer;
    activeTriangles: Uint32Array;
    /** Triangles with flux > 0 (activeTriangles is padded to one element when there are none). */
    activeTriangleCount: number;
    triToActiveMapping: Uint32Array;
    /** MeshLightData[], 4 uints each. */
    meshData: Uint32Array;
    perMeshInstanceOffset: Uint32Array;
}

/**
 * Per-triangle texture-averaged radiance by (mesh indices, emissive texels): it depends on the UVs only, so animated
 * rebuilds reuse it (native integrates once and only updates positions per frame). NaN = not integrated yet.
 */
const averageCache = new WeakMap<Uint32Array, WeakMap<Float32Array, Float32Array>>();

export function buildLightCollection(meshes: SceneMeshDesc[], materials: EmissiveMaterialInfo[]): LightCollectionData {
    interface Tri {
        posW: float3[];
        uv: [number, number][];
        normal: float3;
        area: number;
        materialID: number;
        lightIdx: number;
        average?: { cache: Float32Array; index: number };
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
        // isWorldFrontFaceCW: a mirroring transform reverses the world winding, so the face normal flips.
        const d = world.data;
        const flip = d[0]! * (d[5]! * d[10]! - d[6]! * d[9]!) - d[1]! * (d[4]! * d[10]! - d[6]! * d[8]!) + d[2]! * (d[4]! * d[9]! - d[5]! * d[8]!) < 0 ? -1 : 1;
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
            const normal = len > 0 ? new float3((flip * n.x) / len, (flip * n.y) / len, (flip * n.z) / len) : new float3(0, 0, 1);
            let average: Tri["average"];
            const texels = mat.emissiveTexture?.rgb;
            if (texels) {
                let perTexture = averageCache.get(mesh.indices);
                if (!perTexture) averageCache.set(mesh.indices, (perTexture = new WeakMap()));
                let cache = perTexture.get(texels);
                if (!cache) perTexture.set(texels, (cache = new Float32Array(mesh.indices.length).fill(NaN)));
                average = { cache, index: t };
            }
            tris.push({ posW: p, uv, normal, area, materialID: mesh.materialID, lightIdx, average });
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

        const mat = materials[tri.materialID]!;
        let rad = mat.radiance;
        if (mat.emissiveTexture) {
            const c = tri.average;
            if (c && Number.isNaN(c.cache[c.index]!)) c.cache.set(integrateEmissiveTexture(mat.emissiveTexture, tri.uv), c.index);
            const avg: [number, number, number] = c ? [c.cache[c.index]!, c.cache[c.index + 1]!, c.cache[c.index + 2]!] : integrateEmissiveTexture(mat.emissiveTexture, tri.uv);
            const factor = mat.emissiveFactor ?? 1;
            rad = [avg[0] * factor, avg[1] * factor, avg[2] * factor];
        }
        const flux = (0.2126 * rad[0] + 0.7152 * rad[1] + 0.0722 * rad[2]) * tri.area * Math.PI;
        fv.setFloat32(i * 32, flux, true);
        fv.setFloat32(i * 32 + 16, rad[0], true);
        fv.setFloat32(i * 32 + 20, rad[1], true);
        fv.setFloat32(i * 32 + 24, rad[2], true);
    });

    // LightCollection::updateActiveTriangleList: only triangles with flux > 0 are active; degenerate
    // (zero-area) triangles, e.g. at a sphere's pole, are culled from what the samplers pick from.
    const activeList: number[] = [];
    const mapping = new Uint32Array(Math.max(tris.length, 1)).fill(kInvalidIndex);
    for (let i = 0; i < tris.length; i++) {
        if (fv.getFloat32(i * 32, true) > 0) {
            mapping[i] = activeList.length;
            activeList.push(i);
        }
    }
    const active = new Uint32Array(Math.max(activeList.length, 1));
    active.set(activeList);

    return {
        triangleCount: tris.length,
        meshCount: meshLights.length / 4,
        triangleData,
        fluxData,
        activeTriangles: active,
        activeTriangleCount: activeList.length,
        triToActiveMapping: mapping,
        meshData: meshLights.length > 0 ? new Uint32Array(meshLights) : new Uint32Array([kInvalidIndex, kInvalidIndex, 0, kInvalidIndex]),
        perMeshInstanceOffset: perMeshInstanceOffset.length > 0 ? perMeshInstanceOffset : new Uint32Array([kInvalidIndex]),
    };
}
