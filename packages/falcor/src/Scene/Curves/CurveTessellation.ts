/**
 * Curve tessellation mirroring Scene/Curves/CurveTessellation.cpp and
 * Utils/Math/CubicSpline.h: strands of control points become either
 * linear-swept-sphere segments (convertToLinearSweptSphere) or triangle
 * "poly tubes" (convertToPolytube), both through a natural cubic spline (with
 * USD defaults subdiv=1 this passes the deduped control points through exactly).
 */

import { float4x4, transformPoint } from "../../Utils/Math/Matrix.js";
import { float3, cross, normalize3 } from "../../Utils/Math/Vector.js";
import { quatFromRotationBetweenVectors, rotateVector } from "../../Utils/Math/Quaternion.js";
import { CubicSpline } from "../../Utils/Math/CubicSpline.js";

export { CubicSpline };

/** Smallest normalized fp16 (native sanitizeWidth floor). */
const kMinRadius = 6.103515625e-5;

/** Quad-tube width compensation (native kMeshCompensationScale) — polytube path only. */
export const kMeshCompensationScale = 1.11;

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
    /** Full prim path (e.g. /Root/curve0), what Settings attribute filters match. */
    path: string;
    curveVertexCounts: Uint32Array;
    /** xyz control points, concatenated across strands. */
    points: Float32Array;
    /** Per-vertex widths (diameters, USD convention). */
    widths: Float32Array;
    /** Time-sampled points (time codes >= 1, as native's processCurve keeps); `points` is the first. */
    pointsSamples?: { time: number; points: Float32Array }[];
}

/** Extracts BasisCurves prims from USDA text (tinyusdz's RenderScene API
 *  does not expose curves; binary .usdc curves are unsupported until it does). */
export function extractBasisCurvesFromUsda(source: string): BasisCurvesDesc[] {
    const out: BasisCurvesDesc[] = [];
    const re = /def BasisCurves "([^"]+)"[^{]*\{/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(source)) !== null) {
        const path = primPathAt(source, m.index, m[1]!);
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
        let pts = nums("point3f\\[\\] points");
        // points.timeSamples = { time: [...], ... }; native drops samples with time code < 1.
        let pointsSamples: { time: number; points: Float32Array }[] | undefined;
        const ts = /points\.timeSamples\s*=\s*\{/.exec(body);
        if (ts) {
            let k = ts.index + ts[0].length;
            const from = k;
            for (let d = 1; k < body.length && d > 0; k++) d += body[k] === "{" ? 1 : body[k] === "}" ? -1 : 0;
            pointsSamples = [...body.slice(from, k - 1).matchAll(/([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*:\s*\[([^\]]*)\]/g)]
                .map((e) => ({ time: Number(e[1]), points: new Float32Array((e[2]!.match(/-?[\d.eE+]+/g) ?? []).map(Number)) }))
                .sort((a, b) => a.time - b.time)
                .filter((e) => e.time >= 1);
            if (pointsSamples.length > 0) pts = Array.from(pointsSamples[0]!.points);
            if (pointsSamples.length < 2) pointsSamples = undefined;
        }
        const widths = nums("float\\[\\] widths");
        if (!counts || !pts) continue;
        const vertexTotal = counts.reduce((acc, c) => acc + c, 0);
        out.push({
            name: m[1]!,
            path,
            curveVertexCounts: new Uint32Array(counts),
            points: new Float32Array(pts),
            widths: new Float32Array(widths ?? new Array<number>(vertexTotal).fill(1)),
            pointsSamples,
        });
    }
    return out;
}

/**
 * The prim path of a `def` at `offset`: every `def ... "name" {` still open
 * there is an ancestor, so the path is their names in nesting order.
 */
function primPathAt(source: string, offset: number, name: string): string {
    const stack: (string | null)[] = [];
    const token = /def\s+(?:\w+\s+)?"([^"]+)"[^{]*\{|\{|\}/g;
    let m: RegExpExecArray | null;
    while ((m = token.exec(source)) !== null && m.index < offset) {
        if (m[0] === "}") stack.pop();
        else stack.push(m[1] ?? null); // bare braces (metadata, dictionaries) nest too
    }
    return "/" + [...stack.filter((n): n is string => n !== null), name].join("/");
}

/** Mirrors perp_stark (MathHelpers.h): a unit vector perpendicular to u. */
function perpStark(u: float3): float3 {
    const ax = Math.abs(u.x);
    const ay = Math.abs(u.y);
    const az = Math.abs(u.z);
    const uyx = ax - ay < 0 ? 1 : 0;
    const uzx = ax - az < 0 ? 1 : 0;
    const uzy = ay - az < 0 ? 1 : 0;
    const xm = uyx & uzx;
    const ym = (1 ^ xm) & uzy;
    const zm = 1 ^ (xm | ym);
    return normalize3(cross(u, new float3(xm, ym, zm)));
}

export interface PolytubeMeshResult {
    /** xyz per vertex. */
    vertices: Float32Array;
    normals: Float32Array;
    /** xyzw per vertex: the curve's forward direction, sign 1. */
    tangents: Float32Array;
    /** uv per vertex, or null when the curves carry none. */
    texCrds: Float32Array | null;
    /** Tube radius at each vertex (non-zero marks mesh-from-curves). */
    radii: Float32Array;
    /** Triangle list. */
    faceVertexIndices: Uint32Array;
}

/** Native optimizeStrandGeometry: dedup, then resample through the splines. */
function optimizeStrandGeometry(
    controlPoints: ArrayLike<number>,
    widths: ArrayLike<number>,
    uvs: ArrayLike<number> | null,
    pointOffset: number,
    vertexCount: number,
    subdivPerSegment: number,
    keepOneEveryXVerticesPerStrand: number,
    widthScale: number,
): { points: float3[]; widths: number[]; uvs: [number, number][] | null } {
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
    const sanitize = (w: number) => Math.max(w, kMinRadius);
    const outPoints: float3[] = [];
    const outWidths: number[] = [];
    const sample = (j: number, t: number) => {
        outPoints.push(new float3(splinePoints.interpolate(j, t, 0), splinePoints.interpolate(j, t, 1), splinePoints.interpolate(j, t, 2)));
        outWidths.push(sanitize(kMeshCompensationScale * widthScale * splineWidths.interpolate(j, t, 0)));
    };
    let tmpCount = 0;
    for (let j = 0; j < n - 1; j++) {
        for (let k = 0; k < subdivPerSegment; k++) {
            if (tmpCount % keepOneEveryXVerticesPerStrand === 0) sample(j, k / subdivPerSegment);
            tmpCount++;
        }
    }
    sample(n - 2, 1); // always keep the last vertex

    let outUVs: [number, number][] | null = null;
    if (uvs) {
        const splineUVs = new CubicSpline(uv, n, 2);
        outUVs = [];
        tmpCount = 0;
        for (let j = 0; j < n - 1; j++) {
            for (let k = 0; k < subdivPerSegment; k++) {
                if (tmpCount % keepOneEveryXVerticesPerStrand === 0) {
                    const t = k / subdivPerSegment;
                    outUVs.push([splineUVs.interpolate(j, t, 0), splineUVs.interpolate(j, t, 1)]);
                }
                tmpCount++;
            }
        }
        outUVs.push([splineUVs.interpolate(n - 2, 1, 0), splineUVs.interpolate(n - 2, 1, 1)]);
    }
    return { points: outPoints, widths: outWidths, uvs: outUVs };
}

/**
 * Mirrors CurveTessellation::convertToPolytube: each strand becomes a tube of
 * `pointCountPerCrossSection`-gon cross sections, carried along the curve by a
 * rotation-minimizing frame. Widths are pre-scaled by kMeshCompensationScale
 * so the tube reads, on average over view angles, as wide as the curve.
 */
export function convertToPolytube(
    strandCount: number,
    vertexCountsPerStrand: ArrayLike<number>,
    controlPoints: ArrayLike<number>,
    widths: ArrayLike<number>,
    uvs: ArrayLike<number> | null,
    subdivPerSegment: number,
    keepOneEveryXStrands: number,
    keepOneEveryXVerticesPerStrand: number,
    widthScale: number,
    pointCountPerCrossSection: number,
): PolytubeMeshResult {
    const vertices: number[] = [];
    const normals: number[] = [];
    const tangents: number[] = [];
    const texCrds: number[] = [];
    const radii: number[] = [];
    const faces: number[] = [];
    const P = pointCountPerCrossSection;
    let pointOffset = 0;
    let meshVertexOffset = 0;

    for (let i = 0; i < strandCount; i += keepOneEveryXStrands) {
        const strand = optimizeStrandGeometry(controlPoints, widths, uvs, pointOffset, vertexCountsPerStrand[i]!, subdivPerSegment, keepOneEveryXVerticesPerStrand, widthScale);
        for (let j = i; j < Math.min(strandCount, i + keepOneEveryXStrands); j++) pointOffset += vertexCountsPerStrand[j]!;
        const pts = strand.points;
        const sub = (a: float3, b: float3) => new float3(a.x - b.x, a.y - b.y, a.z - b.z);

        // Initial frame (buildFrame): s = perp_stark(fwd), t = fwd x s.
        let fwd = normalize3(sub(pts[1]!, pts[0]!));
        let s = perpStark(fwd);
        let t = cross(fwd, s);

        for (let j = 0; j < pts.length; j++) {
            // updateCurveFrame: advance fwd, then rotate s by the change in fwd.
            let prevFwd = fwd;
            if (j <= 0 || j >= pts.length || pts.length === 2) {
                prevFwd = fwd;
            } else if (j === 1) {
                prevFwd = normalize3(sub(pts[j]!, pts[j - 1]!));
                fwd = normalize3(sub(pts[j + 1]!, pts[j - 1]!));
            } else if (j < pts.length - 2) {
                prevFwd = normalize3(sub(pts[j]!, pts[j - 2]!));
                fwd = normalize3(sub(pts[j + 1]!, pts[j - 1]!));
            } else if (j === pts.length - 1) {
                prevFwd = normalize3(sub(pts[j]!, pts[j - 2]!));
                fwd = normalize3(sub(pts[j]!, pts[j - 1]!));
            }
            s = rotateVector(quatFromRotationBetweenVectors(prevFwd, fwd), s);
            t = normalize3(cross(fwd, s));
            s = normalize3(cross(t, fwd));

            // updateMeshResultBuffers: one ring of P vertices around the point.
            const r = 0.5 * strand.widths[j]!;
            for (let k = 0; k < P; k++) {
                const phi = (k / P) * Math.PI * 2;
                const nx = Math.cos(phi) * s.x + Math.sin(phi) * t.x;
                const ny = Math.cos(phi) * s.y + Math.sin(phi) * t.y;
                const nz = Math.cos(phi) * s.z + Math.sin(phi) * t.z;
                vertices.push(pts[j]!.x + r * nx, pts[j]!.y + r * ny, pts[j]!.z + r * nz);
                normals.push(nx, ny, nz);
                tangents.push(fwd.x, fwd.y, fwd.z, 1);
                radii.push(r);
                if (strand.uvs) texCrds.push(strand.uvs[j]![0], strand.uvs[j]![1]);
            }

            // connectFaceVertices: two triangles per quad to the next ring.
            if (j < pts.length - 1) {
                for (let k = 0; k < P; k++) {
                    const a = meshVertexOffset + j * P + k;
                    const b = meshVertexOffset + j * P + ((k + 1) % P);
                    const c = meshVertexOffset + (j + 1) * P + ((k + 1) % P);
                    const d = meshVertexOffset + (j + 1) * P + k;
                    faces.push(a, b, c, a, c, d);
                }
            }
        }
        meshVertexOffset += P * pts.length;
    }

    return {
        vertices: new Float32Array(vertices),
        normals: new Float32Array(normals),
        tangents: new Float32Array(tangents),
        texCrds: uvs ? new Float32Array(texCrds) : null,
        radii: new Float32Array(radii),
        faceVertexIndices: new Uint32Array(faces),
    };
}
