/**
 * SDF 3D primitives mirroring Scene/SDFs/SDF3DPrimitive{,Common}.slang and the
 * `.sdf` files `SDFGrid::loadPrimitivesFromFile` reads (a JSON array of
 * primitives, keys as `SDFGrid.cpp`'s `to_json`/`from_json` name them).
 *
 * The evaluator mirrors `EvaluateSDFPrimitives.cs.slang` value for value:
 * sample position `p = -0.5 + coords / gridWidth`, distances folded together in
 * array order starting from FLT_MAX, written x-fastest. Web divergence (docs
 * §9): native runs that kernel on the GPU and only SDFSBS accepts primitives at
 * all — the port builds SDF grids host-side, so the same loop runs here and its
 * corner values feed the ordinary `setValues` path, which every grid type takes.
 */

import { RuntimeError } from "../../Core/Error.js";

/** Mirrors SDF3DShapeType; the strings are the `.sdf` JSON spellings. */
export enum SDF3DShapeType {
    Sphere = 0,
    Ellipsoid = 1,
    Box = 2,
    Torus = 3,
    Cone = 4,
    Capsule = 5,
}

/** Mirrors SDFOperationType. */
export enum SDFOperationType {
    Union = 0,
    Subtraction = 1,
    Intersection = 2,
    SmoothUnion = 3,
    SmoothSubtraction = 4,
    SmoothIntersection = 5,
}

const kShapeNames = ["sphere", "ellipsoid", "box", "torus", "cone", "capsule"];
const kOperationNames = ["union", "subtraction", "intersection", "smooth_union", "smooth_subtraction", "smooth_intersection"];

export interface SDF3DPrimitive {
    shapeType: SDF3DShapeType;
    /** Shape parameters; how many components are used depends on the type. */
    shapeData: [number, number, number];
    shapeBlobbing: number;
    operationType: SDFOperationType;
    operationSmoothing: number;
    translation: [number, number, number];
    /** Inverted rotation+scale, row-major (9 floats), as the JSON stores it. */
    invRotationScale: number[];
}

/** Largest float32, the value the evaluation kernel starts from. */
const kFltMax = 3.402823466e38;

function lookupEnum(value: unknown, names: string[], what: string): number {
    // Native's from_json accepts both the string spelling and the old uint form.
    if (typeof value === "number") return value;
    const index = names.indexOf(String(value));
    if (index < 0) throw new RuntimeError(`SDF primitive: unknown ${what} '${String(value)}'`);
    return index;
}

function toVec3(value: unknown, what: string): [number, number, number] {
    if (!Array.isArray(value) || value.length !== 3) throw new RuntimeError(`SDF primitive: '${what}' must be an array of 3 numbers`);
    return [Number(value[0]), Number(value[1]), Number(value[2])];
}

/** Mirrors SDFGrid::loadPrimitivesFromFile's JSON deserialization. */
export function parseSDFPrimitives(text: string): SDF3DPrimitive[] {
    const doc = JSON.parse(text) as unknown;
    if (!Array.isArray(doc)) throw new RuntimeError("SDF primitive file: expected a JSON array of primitives");
    return doc.map((entry) => {
        const j = entry as Record<string, unknown>;
        const m = j["inv_rot_scale"];
        if (!Array.isArray(m) || m.length !== 9) throw new RuntimeError("SDF primitive: 'inv_rot_scale' must be an array of 9 numbers");
        return {
            shapeType: lookupEnum(j["shape_type"], kShapeNames, "shape type"),
            shapeData: toVec3(j["shape_data"], "shape_data"),
            shapeBlobbing: Number(j["shape_blobbing"] ?? 0),
            operationType: lookupEnum(j["operation_type"], kOperationNames, "operation type"),
            operationSmoothing: Number(j["operation_smoothing"] ?? 0),
            translation: toVec3(j["translation"], "translation"),
            invRotationScale: m.map(Number),
        };
    });
}

/** Mirrors SDFGrid::writePrimitivesToFile (nlohmann's 4-space dump). */
export function serializeSDFPrimitives(primitives: readonly SDF3DPrimitive[]): string {
    return JSON.stringify(
        primitives.map((p) => ({
            shape_type: kShapeNames[p.shapeType],
            shape_data: p.shapeData,
            shape_blobbing: p.shapeBlobbing,
            operation_type: kOperationNames[p.operationType],
            operation_smoothing: p.operationSmoothing,
            translation: p.translation,
            inv_rot_scale: p.invRotationScale,
        })),
        null,
        4,
    );
}

const length3 = (x: number, y: number, z: number) => Math.sqrt(x * x + y * y + z * z);
const saturate = (v: number) => Math.min(Math.max(v, 0), 1);

/** Mirrors Utils/SDF/SDF3DShapes.slang for a point already in shape space. */
function evalShapeLocal(px: number, py: number, pz: number, type: SDF3DShapeType, data: readonly number[]): number {
    switch (type) {
        case SDF3DShapeType.Sphere:
            return length3(px, py, pz) - data[0]!;
        case SDF3DShapeType.Ellipsoid: {
            const [rx, ry, rz] = [data[0]!, data[1]!, data[2]!];
            const k0 = length3(px / rx, py / ry, pz / rz);
            const k1 = length3(px / (rx * rx), py / (ry * ry), pz / (rz * rz));
            return (k0 * (k0 - 1)) / k1;
        }
        case SDF3DShapeType.Box: {
            const qx = Math.abs(px) - data[0]!;
            const qy = Math.abs(py) - data[1]!;
            const qz = Math.abs(pz) - data[2]!;
            return length3(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
        }
        case SDF3DShapeType.Torus: {
            // The tube has zero radius here; thickness comes from blobbing.
            const d = Math.hypot(px, pz) - data[0]!;
            return Math.hypot(d, py);
        }
        case SDF3DShapeType.Cone: {
            const h = data[1]!;
            const qx = h * data[0]!;
            const qy = -h;
            const wx = Math.hypot(px, pz);
            const wy = py - 0.5 * h;
            const t = saturate((wx * qx + wy * qy) / (qx * qx + qy * qy));
            const ax = wx - qx * t;
            const ay = wy - qy * t;
            const bx = wx - qx * saturate(wx / qx);
            const by = wy - qy;
            const k = Math.sign(qy);
            const d = Math.min(ax * ax + ay * ay, bx * bx + by * by);
            const s = Math.max(k * (wx * qy - wy * qx), k * (wy - qy));
            return Math.sqrt(d) * Math.sign(s);
        }
        case SDF3DShapeType.Capsule: {
            const hl = data[0]!;
            const y = py - Math.min(Math.max(py, -hl), hl);
            return length3(px, y, pz);
        }
        default:
            throw new RuntimeError(`SDF primitive: unknown shape type ${type}`);
    }
}

/** Mirrors Utils/SDF/SDFOperations.slang's smin/smax (k == 0 is degenerate there too). */
const smin = (a: number, b: number, k: number) => {
    const h = Math.max(k - Math.abs(a - b), 0);
    return Math.min(a, b) - (h * h * 0.25) / k;
};
const smax = (a: number, b: number, k: number) => {
    const h = Math.max(k - Math.abs(a - b), 0);
    return Math.max(a, b) + (h * h * 0.25) / k;
};

/** Mirrors SDF3DPrimitive::evalOperation. */
function evalOperation(type: SDFOperationType, d: number, dShape: number, smoothing: number): number {
    switch (type) {
        case SDFOperationType.Union:
            return Math.min(d, dShape);
        case SDFOperationType.Subtraction:
            return Math.max(d, -dShape);
        case SDFOperationType.Intersection:
            return Math.max(d, dShape);
        case SDFOperationType.SmoothUnion:
            return smin(d, dShape, smoothing);
        case SDFOperationType.SmoothSubtraction:
            return smax(d, -dShape, smoothing);
        case SDFOperationType.SmoothIntersection:
            return smax(d, dShape, smoothing);
        default:
            throw new RuntimeError(`SDF primitive: unknown operation type ${type}`);
    }
}

/** Mirrors SDF3DPrimitive::eval(p, d): the shape folded into the running distance. */
export function evalSDFPrimitive(primitive: SDF3DPrimitive, p: readonly [number, number, number], d: number): number {
    const m = primitive.invRotationScale;
    const t = primitive.translation;
    const [vx, vy, vz] = [p[0] - t[0], p[1] - t[1], p[2] - t[2]];
    // mul(transpose(invRotationScale), p - translation) with a row-major matrix.
    const lx = m[0]! * vx + m[3]! * vy + m[6]! * vz;
    const ly = m[1]! * vx + m[4]! * vy + m[7]! * vz;
    const lz = m[2]! * vx + m[5]! * vy + m[8]! * vz;
    const dShape = evalShapeLocal(lx, ly, lz, primitive.shapeType, primitive.shapeData) - primitive.shapeBlobbing;
    return evalOperation(primitive.operationType, d, dShape, primitive.operationSmoothing);
}

/**
 * Mirrors EvaluateSDFPrimitives.cs.slang: evaluates the primitive list at every
 * grid corner of a `gridWidth` grid spanning [-0.5, 0.5]^3. With `base` (a grid
 * loaded from values), the primitives fold into those values instead of empty space.
 *
 * @returns (gridWidth + 1)^3 corner values, x-fastest (flatten3D order).
 */
export function evaluateSDFPrimitives(primitives: readonly SDF3DPrimitive[], gridWidth: number, base?: Float32Array): Float32Array {
    const w = gridWidth + 1;
    const values = new Float32Array(w * w * w);
    for (let z = 0; z < w; z++) {
        for (let y = 0; y < w; y++) {
            for (let x = 0; x < w; x++) {
                const p: [number, number, number] = [-0.5 + x / gridWidth, -0.5 + y / gridWidth, -0.5 + z / gridWidth];
                let d = base ? base[x + w * (y + w * z)]! : kFltMax;
                for (const primitive of primitives) d = evalSDFPrimitive(primitive, p, d);
                values[x + w * (y + w * z)] = d;
            }
        }
    }
    return values;
}

/** Fetches and parses a `.sdf` primitive file (docs §9: asset IO is async). */
export async function loadSDFPrimitives(url: string): Promise<SDF3DPrimitive[]> {
    const response = await fetch(url);
    if (!response.ok) throw new RuntimeError(`SDF primitive file: failed to fetch '${url}' (${response.status})`);
    return parseSDFPrimitives(await response.text());
}
