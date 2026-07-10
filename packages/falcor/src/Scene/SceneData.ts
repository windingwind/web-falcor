/**
 * Host-side packers for GPU scene structures (Scene/SceneTypes.slang layouts
 * transcribed from upstream).
 */

import { float2, float3, float4, length3 } from "../Utils/Math/Vector.js";
import { float4x4, inverse, matrixFromRotationAxisAngle, transformPoint, transformVector, transpose } from "../Utils/Math/Matrix.js";
import { f32tof16 } from "./Material/MaterialData.js";

export interface StaticVertex {
    position: float3;
    normal: float3;
    /** xyz = tangent, w = sign (0 if invalid). */
    tangent: float4;
    texCrd: float2;
    curveRadius?: number;
}

/** Octahedral snorm2x16 encode (Utils/Math/PackedFormats.slang encodeNormal2x16). */
export function encodeNormal2x16(n: float3): number {
    const l1 = Math.abs(n.x) + Math.abs(n.y) + Math.abs(n.z) || 1;
    let ox = n.x / l1;
    let oy = n.y / l1;
    if (n.z < 0) {
        const tx = (1 - Math.abs(oy)) * (ox >= 0 ? 1 : -1);
        const ty = (1 - Math.abs(ox)) * (oy >= 0 ? 1 : -1);
        ox = tx;
        oy = ty;
    }
    const sx = Math.round(Math.min(Math.max(ox, -1), 1) * 32767) & 0xffff;
    const sy = Math.round(Math.min(Math.max(oy, -1), 1) * 32767) & 0xffff;
    return (sy << 16) | sx;
}

/**
 * WGSL address-space layout: vec3 members align to 16 bytes, so the packed
 * vertex struct is 48 bytes on the web (native Falcor packs it into 32).
 * Element offsets: position@0, packedNormalTangentCurveRadius@16, texCrd@32.
 */
export const kPackedStaticVertexSize = 48;

/** Packs vertices into PackedStaticVertexData layout (SceneTypes.slang pack()). */
export function packStaticVertices(vertices: StaticVertex[]): ArrayBuffer {
    const buffer = new ArrayBuffer(vertices.length * kPackedStaticVertexSize);
    const dv = new DataView(buffer);
    for (let i = 0; i < vertices.length; i++) {
        const v = vertices[i]!;
        const base = i * kPackedStaticVertexSize;
        dv.setFloat32(base + 0, v.position.x, true);
        dv.setFloat32(base + 4, v.position.y, true);
        dv.setFloat32(base + 8, v.position.z, true);

        const nx = f32tof16(v.normal.x);
        const ny = f32tof16(v.normal.y);
        const nz = f32tof16(v.normal.z);
        let packedTangentSign = v.tangent.w;
        if ((v.curveRadius ?? 0) > 0) packedTangentSign *= v.curveRadius!;
        const tw = f32tof16(packedTangentSign);

        dv.setUint32(base + 16, ((ny << 16) | nx) >>> 0, true);
        dv.setUint32(base + 20, ((tw << 16) | nz) >>> 0, true);
        dv.setUint32(base + 24, encodeNormal2x16(new float3(v.tangent.x, v.tangent.y, v.tangent.z)) >>> 0, true);

        dv.setFloat32(base + 32, v.texCrd.x, true);
        dv.setFloat32(base + 36, v.texCrd.y, true);
    }
    return buffer;
}

/** Packs PrevVertexData (float3 position; WGSL std430 stride 16). */
export function packPrevVertices(vertices: { position: { x: number; y: number; z: number } }[]): Float32Array {
    const out = new Float32Array(vertices.length * 4);
    for (let i = 0; i < vertices.length; i++) {
        const p = vertices[i]!.position;
        out[i * 4] = p.x;
        out[i * 4 + 1] = p.y;
        out[i * 4 + 2] = p.z;
    }
    return out;
}

/** Mirrors GeometryType (SceneDefines.slangh GEOMETRY_TYPE_* values). */
export enum GeometryType {
    None = 0,
    TriangleMesh = 1,
    DisplacedTriangleMesh = 2,
    Curve = 3,
    SDFGrid = 5,
    Custom = 6,
}

export const kGeometryInstanceSize = 32; // 8 uints

export interface GeometryInstance {
    type: GeometryType;
    globalMatrixID: number;
    materialID: number;
    geometryID: number;
    vbOffset: number;
    ibOffset: number;
    instanceIndex: number;
    geometryIndex: number;
    flags?: number;
}

export function packGeometryInstances(instances: GeometryInstance[]): Uint32Array {
    const out = new Uint32Array(instances.length * 8);
    for (let i = 0; i < instances.length; i++) {
        const inst = instances[i]!;
        // Upper 3 bits of flags store the geometry type (kTypeOffset = 29).
        out[i * 8 + 0] = (((inst.type & 0x7) << 29) | ((inst.flags ?? 0) & 0x1fffffff)) >>> 0;
        out[i * 8 + 1] = inst.globalMatrixID;
        out[i * 8 + 2] = inst.materialID;
        out[i * 8 + 3] = inst.geometryID;
        out[i * 8 + 4] = inst.vbOffset;
        out[i * 8 + 5] = inst.ibOffset;
        out[i * 8 + 6] = inst.instanceIndex;
        out[i * 8 + 7] = inst.geometryIndex;
    }
    return out;
}

export const kMeshDescSize = 32; // 8 uints

export interface MeshDescData {
    vbOffset: number;
    ibOffset: number;
    vertexCount: number;
    indexCount: number;
    skinningVbOffset?: number;
    prevVbOffset?: number;
    materialID: number;
    flags?: number; // MeshFlags: bit0 = Use16BitIndices, bit1 = IsSkinned, ...
}

export function packMeshDescs(meshes: MeshDescData[]): Uint32Array {
    const out = new Uint32Array(meshes.length * 8);
    for (let i = 0; i < meshes.length; i++) {
        const m = meshes[i]!;
        out.set(
            [m.vbOffset, m.ibOffset, m.vertexCount, m.indexCount, m.skinningVbOffset ?? 0, m.prevVbOffset ?? 0, m.materialID, m.flags ?? 0],
            i * 8,
        );
    }
    return out;
}

/** Mirrors LightType (LightData.slang). */
export enum LightType {
    Point = 0,
    Directional = 1,
    Distant = 2,
    Rect = 3,
    Disc = 4,
    Sphere = 5,
}

export interface AnalyticLight {
    type: LightType;
    /** Light name (pyscene ctor arg); used by Scene.getLight(name). */
    name?: string;
    /** Position (point lights) in world space. */
    posW?: float3;
    /** Direction (directional/spot), normalized. */
    dirW?: float3;
    /** Emitted radiance/intensity. */
    intensity: float3;
    /** Distant light: half-angle subtended by the light (radians). */
    angle?: number;
    /** Point/spot light: cutoff half-angle (radians); PI = omnidirectional point. */
    openingAngle?: number;
    /** Point/spot light: penumbra half-angle (radians) for soft cutoff. */
    penumbraAngle?: number;
    /** Area lights (Rect/Disc/Sphere): local->world transform incl. scaling.
     *  Local shapes live in the z=0 xy-plane ([-1,1]^2), unit disc, or unit sphere. */
    transMat?: float4x4;
    /** Animated scenes: node whose global matrix drives this light each frame
     *  (Animatable::updateFromAnimation). Position/direction/transform follow it. */
    nodeID?: number;
}

export const kLightDataSize = 224; // 6x16B rows + 2x 64B float4x4 (rows at 96/160)

/** Packs LightData (LightData.slang layout; 16-byte rows, matrices identity). */
export function packLights(lights: AnalyticLight[]): ArrayBuffer {
    const buffer = new ArrayBuffer(Math.max(lights.length, 1) * kLightDataSize);
    const dv = new DataView(buffer);
    lights.forEach((light, i) => {
        const base = i * kLightDataSize;
        const posW = light.posW ?? new float3(0, 0, 0);
        let dirW = light.dirW ?? new float3(0, -1, 0);
        if (light.type === LightType.Directional || light.type === LightType.Distant) {
            // Native setWorldDirection normalizes.
            const len = Math.hypot(dirW.x, dirW.y, dirW.z) || 1;
            dirW = new float3(dirW.x / len, dirW.y / len, dirW.z / len);
        }
        const isArea = light.type === LightType.Rect || light.type === LightType.Disc || light.type === LightType.Sphere;
        // Area lights derive posW from the transform's translation (shader samples
        // the shape via transMat; posW is only used for culling/UI).
        const areaMat = light.transMat ?? float4x4.identity();
        const areaPos = isArea ? transformPoint(areaMat, new float3(0, 0, 0)) : posW;
        dv.setFloat32(base + 0, areaPos.x, true);
        dv.setFloat32(base + 4, areaPos.y, true);
        dv.setFloat32(base + 8, areaPos.z, true);
        dv.setUint32(base + 12, light.type, true);
        dv.setFloat32(base + 16, dirW.x, true);
        dv.setFloat32(base + 20, dirW.y, true);
        dv.setFloat32(base + 24, dirW.z, true);
        // openingAngle: PI (full sphere) unless a spot cutoff is given.
        const openingAngle = light.type === LightType.Point && light.openingAngle !== undefined ? light.openingAngle : Math.PI;
        dv.setFloat32(base + 28, openingAngle, true);
        dv.setFloat32(base + 32, light.intensity.x, true);
        dv.setFloat32(base + 36, light.intensity.y, true);
        dv.setFloat32(base + 40, light.intensity.z, true);
        dv.setFloat32(base + 44, Math.cos(openingAngle), true); // cosOpeningAngle
        // cosSubtendedAngle: distant lights use cos(half-angle); default = sun.
        const cosSubtended = light.type === LightType.Distant && light.angle !== undefined ? Math.cos(light.angle) : 0.9999893;
        dv.setFloat32(base + 48, cosSubtended, true);
        dv.setFloat32(base + 52, light.penumbraAngle ?? 0, true); // penumbraAngle

        // Area-light shape params (tangent/bitangent are the local axes; surfaceArea
        // per Light.cpp {Rect,Disc,Sphere}Light::update using the transformed axes).
        if (isArea) {
            const rx = length3(transformVector(areaMat, new float3(1, 0, 0)));
            const ry = length3(transformVector(areaMat, new float3(0, 1, 0)));
            const rz = length3(transformVector(areaMat, new float3(0, 0, 1)));
            let surfaceArea = 4 * rx * ry;
            if (light.type === LightType.Disc) surfaceArea = Math.PI * rx * ry;
            else if (light.type === LightType.Sphere)
                surfaceArea = 4 * Math.PI * Math.pow(Math.pow(rx * ry, 1.6) + Math.pow(ry * rz, 1.6) + Math.pow(rx * rz, 1.6) / 3, 1 / 1.6);
            dv.setFloat32(base + 64, 1, true); // tangent = (1,0,0)
            dv.setFloat32(base + 68, 0, true);
            dv.setFloat32(base + 72, 0, true);
            dv.setFloat32(base + 76, surfaceArea, true);
            dv.setFloat32(base + 80, 0, true); // bitangent = (0,1,0)
            dv.setFloat32(base + 84, 1, true);
            dv.setFloat32(base + 88, 0, true);
        }

        // transMat/transMatIT at 96/160. Area lights use the shape's local->world
        // matrix; distant lights orient the sampling disk (DistantLight::update):
        // rotation aligning up=(0,0,1) with -dirW.
        let transMat = isArea ? areaMat : float4x4.identity();
        if (light.type === LightType.Distant) {
            const nd = [-dirW.x, -dirW.y, -dirW.z];
            const len = Math.hypot(nd[0]!, nd[1]!, nd[2]!) || 1;
            const d = [nd[0]! / len, nd[1]! / len, nd[2]! / len];
            const axis = [0 * d[2]! - 1 * d[1]!, 1 * d[0]! - 0 * d[2]!, 0 * d[1]! - 0 * d[0]!]; // cross(up=(0,0,1), d)
            const sinTheta = Math.hypot(axis[0]!, axis[1]!, axis[2]!);
            if (sinTheta > 0) {
                const cosTheta = d[2]!; // dot(up, d)
                transMat = matrixFromRotationAxisAngle(Math.acos(cosTheta), new float3(axis[0]!, axis[1]!, axis[2]!));
            }
        }
        const transMatIT = inverse(transpose(transMat));
        const writeMat = (offset: number, m: float4x4) => {
            const a = m.toArray();
            for (let k = 0; k < 16; k++) dv.setFloat32(base + offset + k * 4, a[k]!, true);
        };
        writeMat(96, transMat);
        writeMat(160, transMatIT);
    });
    return buffer;
}
