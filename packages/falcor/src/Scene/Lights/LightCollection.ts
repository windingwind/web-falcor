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

    const clipArea = (tx: number, ty: number): number => {
        // Sutherland-Hodgman: clip the triangle against the texel square.
        let poly = px.map((p) => [p[0]!, p[1]!]);
        const clipEdge = (inside: (p: number[]) => boolean, intersect: (a: number[], b: number[]) => number[]) => {
            const out: number[][] = [];
            for (let i = 0; i < poly.length; i++) {
                const a = poly[i]!;
                const b = poly[(i + 1) % poly.length]!;
                const ain = inside(a);
                const bin = inside(b);
                if (ain) out.push(a);
                if (ain !== bin) out.push(intersect(a, b));
            }
            poly = out;
        };
        const lerpAt = (a: number[], b: number[], t: number) => [a[0]! + (b[0]! - a[0]!) * t, a[1]! + (b[1]! - a[1]!) * t];
        clipEdge((p) => p[0]! >= tx, (a, b) => lerpAt(a, b, (tx - a[0]!) / (b[0]! - a[0]!)));
        clipEdge((p) => p[0]! <= tx + 1, (a, b) => lerpAt(a, b, (tx + 1 - a[0]!) / (b[0]! - a[0]!)));
        clipEdge((p) => p[1]! >= ty, (a, b) => lerpAt(a, b, (ty - a[1]!) / (b[1]! - a[1]!)));
        clipEdge((p) => p[1]! <= ty + 1, (a, b) => lerpAt(a, b, (ty + 1 - a[1]!) / (b[1]! - a[1]!)));
        let area = 0;
        for (let i = 0; i < poly.length; i++) {
            const a = poly[i]!;
            const b = poly[(i + 1) % poly.length]!;
            area += a[0]! * b[1]! - b[0]! * a[1]!;
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
            const t = texel(tx, ty);
            sr += a * t[0];
            sg += a * t[1];
            sb += a * t[2];
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

        const mat = materials[tri.materialID]!;
        let rad = mat.radiance;
        if (mat.emissiveTexture) {
            const avg = integrateEmissiveTexture(mat.emissiveTexture, tri.uv);
            const factor = mat.emissiveFactor ?? 1;
            rad = [avg[0] * factor, avg[1] * factor, avg[2] * factor];
        }
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
