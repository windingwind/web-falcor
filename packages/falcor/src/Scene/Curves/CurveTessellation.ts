/**
 * Curve tessellation mirroring Scene/Curves/CurveTessellation.cpp
 * (convertToLinearSweptSphere) and Utils/Math/CubicSpline.h: strands of
 * control points become linear-swept-sphere segments through a natural
 * cubic spline (with USD defaults subdiv=1 this passes the deduped control
 * points through exactly).
 */

import { float4x4, transformPoint } from "../../Utils/Math/Matrix.js";
import { float3 } from "../../Utils/Math/Vector.js";

/** Smallest normalized fp16 (native sanitizeWidth floor). */
const kMinRadius = 6.103515625e-5;

/** Quad-tube width compensation (native kMeshCompensationScale) — polytube path only. */
export const kMeshCompensationScale = 1.11;

/** Natural cubic spline over N-component lanes (CubicSpline.h setup/interpolate). */
export class CubicSpline {
    private a: Float32Array;
    private b: Float32Array;
    private c: Float32Array;
    private d: Float32Array;
    private readonly lanes: number;

    constructor(controlPoints: ArrayLike<number>, pointCount: number, lanes: number) {
        this.lanes = lanes;
        const n = pointCount;
        this.a = new Float32Array(n * lanes);
        this.b = new Float32Array(n * lanes);
        this.c = new Float32Array(n * lanes);
        this.d = new Float32Array(n * lanes);
        const gamma = new Float32Array(n);
        const delta = new Float32Array(n * lanes);
        const D = new Float32Array(n * lanes);

        gamma[0] = 0.5;
        for (let i = 1; i < n - 1; i++) gamma[i] = 1 / (4 - gamma[i - 1]!);
        gamma[n - 1] = 1 / (2 - gamma[n - 2]!);

        for (let l = 0; l < lanes; l++) {
            delta[l] = 3 * (controlPoints[lanes + l]! - controlPoints[l]!) * gamma[0]!;
        }
        for (let i = 1; i < n; i++) {
            const index = i === n - 1 ? i : i + 1;
            for (let l = 0; l < lanes; l++) {
                delta[i * lanes + l] = (3 * (controlPoints[index * lanes + l]! - controlPoints[(i - 1) * lanes + l]!) - delta[(i - 1) * lanes + l]!) * gamma[i]!;
            }
        }
        for (let l = 0; l < lanes; l++) D[(n - 1) * lanes + l] = delta[(n - 1) * lanes + l]!;
        for (let i = n - 2; i >= 0; i--) {
            for (let l = 0; l < lanes; l++) {
                D[i * lanes + l] = delta[i * lanes + l]! - gamma[i]! * D[(i + 1) * lanes + l]!;
            }
        }
        for (let i = 0; i < n - 1; i++) {
            for (let l = 0; l < lanes; l++) {
                const p0 = controlPoints[i * lanes + l]!;
                const p1 = controlPoints[(i + 1) * lanes + l]!;
                this.a[i * lanes + l] = p0;
                this.b[i * lanes + l] = D[i * lanes + l]!;
                this.c[i * lanes + l] = 3 * (p1 - p0) - 2 * D[i * lanes + l]! - D[(i + 1) * lanes + l]!;
                this.d[i * lanes + l] = 2 * (p0 - p1) + D[i * lanes + l]! + D[(i + 1) * lanes + l]!;
            }
        }
    }

    /** Horner evaluation within a section at t in [0,1] (one lane). */
    interpolate(section: number, t: number, lane: number): number {
        const i = section * this.lanes + lane;
        return ((this.d[i]! * t + this.c[i]!) * t + this.b[i]!) * t + this.a[i]!;
    }
}

export interface SweptSphereResult {
    degree: number;
    /** Segment-start indices into points (one per swept-sphere segment). */
    indices: Uint32Array;
    points: float3[];
    radius: Float32Array;
    texCrds: Float32Array | null;
}

/** Mirrors CurveTessellation::convertToLinearSweptSphere (degree 1 only). */
export function convertToLinearSweptSphere(
    strandCount: number,
    vertexCountsPerStrand: ArrayLike<number>,
    controlPoints: ArrayLike<number>,
    widths: ArrayLike<number>,
    uvs: ArrayLike<number> | null,
    degree: number,
    subdivPerSegment: number,
    keepOneEveryXStrands: number,
    keepOneEveryXVerticesPerStrand: number,
    widthScale: number,
    xform: float4x4,
): SweptSphereResult {
    if (degree !== 1) throw new Error("CurveTessellation: only linear tube segments are supported");
    const indices: number[] = [];
    const points: float3[] = [];
    const radius: number[] = [];
    const texCrds: number[] = [];

    // Isotropic radius scale from the transform (native transformSphere).
    const scale = Math.hypot(xform.get(0, 0), xform.get(0, 1), xform.get(0, 2));
    const emit = (x: number, y: number, z: number, r: number): void => {
        const p = transformPoint(xform, new float3(x, y, z));
        points.push(p);
        radius.push(Math.max(r, kMinRadius) * scale);
    };

    let pointOffset = 0;
    for (let s = 0; s < strandCount; s += keepOneEveryXStrands) {
        const vertexCount = vertexCountsPerStrand[s]!;
        // Dedup consecutive duplicate control points (native optimizeStrandGeometry).
        const pts: number[] = [];
        const ws: number[] = [];
        const uv: number[] = [];
        for (let j = 0; j < vertexCount - 1; j++) {
            const o = (pointOffset + j) * 3;
            const o1 = (pointOffset + j + 1) * 3;
            if (controlPoints[o] !== controlPoints[o1] || controlPoints[o + 1] !== controlPoints[o1 + 1] || controlPoints[o + 2] !== controlPoints[o1 + 2]) {
                pts.push(controlPoints[o]!, controlPoints[o + 1]!, controlPoints[o + 2]!);
                ws.push(widths[pointOffset + j]!);
                if (uvs) uv.push(uvs[(pointOffset + j) * 2]!, uvs[(pointOffset + j) * 2 + 1]!);
            }
        }
        const last = pointOffset + vertexCount - 1;
        pts.push(controlPoints[last * 3]!, controlPoints[last * 3 + 1]!, controlPoints[last * 3 + 2]!);
        ws.push(widths[last]!);
        if (uvs) uv.push(uvs[last * 2]!, uvs[last * 2 + 1]!);
        const n = ws.length;

        const splinePoints = new CubicSpline(pts, n, 3);
        const splineWidths = new CubicSpline(ws, n, 1);

        let tmpCount = 0;
        for (let j = 0; j < n - 1; j++) {
            for (let k = 0; k < subdivPerSegment; k++) {
                if (tmpCount % keepOneEveryXVerticesPerStrand === 0) {
                    const t = k / subdivPerSegment;
                    indices.push(points.length);
                    emit(splinePoints.interpolate(j, t, 0), splinePoints.interpolate(j, t, 1), splinePoints.interpolate(j, t, 2), splineWidths.interpolate(j, t, 0) * 0.5 * widthScale);
                }
                tmpCount++;
            }
        }
        emit(splinePoints.interpolate(n - 2, 1, 0), splinePoints.interpolate(n - 2, 1, 1), splinePoints.interpolate(n - 2, 1, 2), splineWidths.interpolate(n - 2, 1, 0) * 0.5 * widthScale);

        if (uvs) {
            const splineUVs = new CubicSpline(uv, n, 2);
            tmpCount = 0;
            for (let j = 0; j < n - 1; j++) {
                for (let k = 0; k < subdivPerSegment; k++) {
                    if (tmpCount % keepOneEveryXVerticesPerStrand === 0) {
                        const t = k / subdivPerSegment;
                        texCrds.push(splineUVs.interpolate(j, t, 0), splineUVs.interpolate(j, t, 1));
                    }
                    tmpCount++;
                }
            }
            texCrds.push(splineUVs.interpolate(n - 2, 1, 0), splineUVs.interpolate(n - 2, 1, 1));
        }

        for (let j = s; j < Math.min(strandCount, s + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j]!;
    }

    return {
        degree,
        indices: new Uint32Array(indices),
        points,
        radius: new Float32Array(radius),
        texCrds: uvs ? new Float32Array(texCrds) : null,
    };
}

export interface BasisCurvesDesc {
    name: string;
    curveVertexCounts: Uint32Array;
    /** xyz control points, concatenated across strands. */
    points: Float32Array;
    /** Per-vertex widths (diameters, USD convention). */
    widths: Float32Array;
}

/** Extracts BasisCurves prims from USDA text (tinyusdz's RenderScene API
 *  does not expose curves; binary .usdc curves are unsupported until it does). */
export function extractBasisCurvesFromUsda(source: string): BasisCurvesDesc[] {
    const out: BasisCurvesDesc[] = [];
    const re = /def BasisCurves "([^"]+)"[^{]*\{/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(source)) !== null) {
        // Capture the prim body up to the matching close brace.
        let depth = 1;
        let i = re.lastIndex;
        while (i < source.length && depth > 0) {
            if (source[i] === "{") depth++;
            else if (source[i] === "}") depth--;
            i++;
        }
        const body = source.slice(re.lastIndex, i);
        const nums = (attr: string): number[] | null => {
            const a = body.match(new RegExp(`${attr}\\s*=\\s*\\[([^\\]]*)\\]`));
            if (!a) return null;
            return (a[1]!.match(/-?[\d.eE+]+/g) ?? []).map(Number);
        };
        const counts = nums("int\\[\\] curveVertexCounts");
        const pts = nums("point3f\\[\\] points");
        const widths = nums("float\\[\\] widths");
        if (!counts || !pts) continue;
        const vertexTotal = counts.reduce((acc, c) => acc + c, 0);
        out.push({
            name: m[1]!,
            curveVertexCounts: new Uint32Array(counts),
            points: new Float32Array(pts),
            widths: new Float32Array(widths ?? new Array<number>(vertexTotal).fill(1)),
        });
    }
    return out;
}
