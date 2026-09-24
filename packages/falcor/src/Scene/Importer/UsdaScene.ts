/**
 * UsdGeom cameras and UsdLux lights from a USD layer's text (tinyusdz's RenderScene exposes
 * neither; its layerToString gives the USDA text of usda/usdc/usdz layers alike). Mirrors
 * USDImporter: the stage root transform is scale(metersPerUnit), rotated -90 degrees about X
 * for Z-up stages; lights take their prim's world transform (Distant: -Z direction, Sphere:
 * radius scaling, Rect: (-w/2, h/2, -1), Disk: (-r, r, -1)), intensity is 2^exposure *
 * intensity * blackbody(colorTemperature) * color, and a DomeLight becomes the environment
 * map. Cameras follow
 * ImporterContext::createCamera: pose from the USD world transform, focal length in mm, the
 * aperture from the f-stop, depth range and focus distance in meters, film width when the
 * horizontal aperture is authored (else the height).
 */

import { float3, normalize3 } from "../../Utils/Math/Vector.js";
import { extractEulerAngleXYZ, float4x4, inverse, matrixFromRotationAxisAngle, matrixFromScaling, matrixFromTranslation, mulMat, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { matrixFromQuat, mulQuat, quatf, slerp } from "../../Utils/Math/Quaternion.js";
import { decomposeTRS } from "../Animation/SceneAnimation.js";
import { LightType, type AnalyticLight } from "../SceneData.js";
import type { SceneMetadata } from "../Scene.js";
import { Logger } from "../../Utils/Logger.js";

export interface UsdaPrim {
    type: string;
    name: string;
    path: string;
    /** The prim's own properties (nested prims removed). */
    body: string;
    children: UsdaPrim[];
    /** Local-to-world, including the stage root transform. */
    world: float4x4;
    /** Local-to-world in USD space (without the root transform). */
    usdWorld: float4x4;
}

export interface UsdaCamera {
    name: string;
    position: float3;
    target: float3;
    up: float3;
    focalLength: number;
    focalDistance: number;
    apertureRadius: number;
    depthRange: [number, number];
    frameWidth?: number;
    frameHeight?: number;
}

export interface UsdaDomeLight {
    name: string;
    /** texture:file as authored (relative to the layer). */
    file: string;
    intensity: number;
    tint: [number, number, number];
    /** Euler XYZ rotation in degrees (EnvMap::setRotation). */
    rotationDeg: [number, number, number];
}

const deg = Math.PI / 180;
const num = (s: string) => (s.match(/[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g) ?? []).map(Number);

/** Reads an attribute value text (`type name = value`), or undefined. Accepts any of `names`. */
function attr(body: string, names: string[]): string | undefined {
    for (const name of names) {
        const escaped = name.replace(/[.:]/g, (c) => `\\${c}`);
        const m = body.match(new RegExp(`^\\s*(?:uniform\\s+|custom\\s+)?[\\w\\[\\]]+\\s+${escaped}\\s*=\\s*(.+)$`, "m"));
        if (m) return m[1]!.trim();
    }
    return undefined;
}

const attrNumber = (body: string, names: string[], fallback: number) => {
    const v = attr(body, names);
    if (v === undefined) return fallback;
    if (/^(true|false)$/.test(v)) return v === "true" ? 1 : 0;
    return num(v)[0] ?? fallback;
};
const attrVector = (body: string, names: string[], fallback: number[]) => {
    const v = attr(body, names);
    return v === undefined ? fallback : num(v);
};

/** The prim's local transform from its xformOpOrder (USD: ops apply right to left). */
function localTransform(body: string): float4x4 {
    const orderText = attr(body, ["xformOpOrder"]);
    if (!orderText) return float4x4.identity();
    const ops = orderText.match(/"([^"]+)"/g)?.map((s) => s.slice(1, -1)) ?? [];
    let m = float4x4.identity();
    for (const op of ops) {
        const invert = op.startsWith("!invert!");
        const name = invert ? op.slice(8) : op;
        const kind = name.split(":")[1] ?? "";
        const v = num(attr(body, [name]) ?? "");
        let t = float4x4.identity();
        if (kind === "translate") t = matrixFromTranslation(new float3(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0));
        else if (kind === "scale") t = matrixFromScaling(new float3(v[0] ?? 1, v[1] ?? 1, v[2] ?? 1));
        else if (kind === "rotateX" || kind === "rotateY" || kind === "rotateZ") {
            const axis = kind === "rotateX" ? new float3(1, 0, 0) : kind === "rotateY" ? new float3(0, 1, 0) : new float3(0, 0, 1);
            t = matrixFromRotationAxisAngle((v[0] ?? 0) * deg, axis);
        } else if (/^rotate[XYZ]{3}$/.test(kind)) {
            // rotateXYZ: X first, then Y, then Z (as column vectors: Rz * Ry * Rx).
            const axes = { X: new float3(1, 0, 0), Y: new float3(0, 1, 0), Z: new float3(0, 0, 1) } as const;
            const letters = kind.slice(6).split("") as ("X" | "Y" | "Z")[];
            const angle = (l: "X" | "Y" | "Z") => (v["XYZ".indexOf(l)] ?? 0) * deg;
            for (const l of letters) t = mulMat(matrixFromRotationAxisAngle(angle(l), axes[l]), t);
        } else if (kind === "orient") {
            t = matrixFromQuat(new quatf(v[1] ?? 0, v[2] ?? 0, v[3] ?? 0, v[0] ?? 1));
        } else if (kind === "transform") {
            // USD matrices are row-vector (translation in the last row): transpose.
            const rows = [0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => v[c * 4 + r] ?? (r === c ? 1 : 0)));
            t = float4x4.fromRows(rows);
        } else {
            Logger.warning(`USDImporter: unsupported xformOp '${op}' ignored.`);
        }
        m = mulMat(m, invert ? inverse(t) : t);
    }
    return m;
}

/** Top-level prim blocks of text[start, end) and the text between them (the enclosing prim's own properties). */
function scanBlocks(text: string, start: number, end: number): { blocks: { type: string; name: string; inner: [number, number] }[]; own: string } {
    const blocks: { type: string; name: string; inner: [number, number] }[] = [];
    let own = "";
    const re = /\b(def|over)\s+(\w+\s+)?"([^"]+)"\s*(\([^)]*\))?\s*\{/g;
    re.lastIndex = start;
    let cursor = start;
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null && m.index < end) {
        own += text.slice(cursor, m.index);
        let depth = 1;
        let i = re.lastIndex;
        while (i < end && depth > 0) {
            const ch = text[i];
            if (ch === '"') {
                i = text.indexOf('"', i + 1) + 1;
                continue;
            }
            if (ch === "{") depth++;
            else if (ch === "}") depth--;
            i++;
        }
        blocks.push({ type: (m[2] ?? "").trim(), name: m[3]!, inner: [re.lastIndex, i - 1] });
        cursor = i;
        re.lastIndex = i;
    }
    own += text.slice(cursor, end);
    return { blocks, own };
}

/** Parses the prim tree (`def`/`over` blocks, brace matched) of USDA text; `root` is the stage root transform. */
export function parseUsdaPrims(text: string, root: float4x4): UsdaPrim[] {
    const build = (start: number, end: number, parentPath: string, parentWorld: float4x4, parentUsd: float4x4): UsdaPrim[] =>
        scanBlocks(text, start, end).blocks.map(({ type, name, inner }) => {
            const { own } = scanBlocks(text, inner[0], inner[1]);
            const local = localTransform(own);
            const path = `${parentPath}/${name}`;
            const world = mulMat(parentWorld, local);
            const usdWorld = mulMat(parentUsd, local);
            return { type, name, path, body: own, world, usdWorld, children: build(inner[0], inner[1], path, world, usdWorld) };
        });
    return build(0, text.length, "", root, float4x4.identity());
}

/** Stage metadata (metersPerUnit defaults to 0.01, as UsdGeomGetStageMetersPerUnit). */
export function usdaStageInfo(text: string): { metersPerUnit: number; upAxis: "Y" | "Z" } {
    const header = text.match(/^#usda[^\n]*\n\s*\(([\s\S]*?)\n\)/)?.[1] ?? "";
    const mpu = header.match(/metersPerUnit\s*=\s*([-+\d.eE]+)/);
    const up = header.match(/upAxis\s*=\s*"(\w)"/);
    return { metersPerUnit: mpu ? Number(mpu[1]) : 0.01, upAxis: up?.[1] === "Z" ? "Z" : "Y" };
}

/** Mirrors the importer's root transform: scale(metersPerUnit), then -90 degrees about X for Z-up stages. */
export function usdStageRootTransform(info: { metersPerUnit: number; upAxis: "Y" | "Z" }): float4x4 {
    let root = matrixFromScaling(new float3(info.metersPerUnit, info.metersPerUnit, info.metersPerUnit));
    if (info.upAxis === "Z") root = mulMat(matrixFromRotationAxisAngle(-90 * deg, new float3(1, 0, 0)), root);
    return root;
}

// UsdLux's blackbody table, 1000 K to 10000 K in 500 K steps, with end knots repeated.
const kBlackbodyRgb = [
    [1.0, 0.02749, 0.0], [1.0, 0.02749, 0.0], [1.0, 0.149664, 0.0], [1.0, 0.256644, 0.008095], [1.0, 0.372033, 0.06745],
    [1.0, 0.476725, 0.153601], [1.0, 0.570376, 0.259196], [1.0, 0.65348, 0.377155], [1.0, 0.726878, 0.501606],
    [1.0, 0.791543, 0.62805], [1.0, 0.848462, 0.753228], [1.0, 0.898581, 0.874905], [1.0, 0.942771, 0.991642],
    [0.906947, 0.890456, 1.0], [0.828247, 0.841838, 1.0], [0.765791, 0.801896, 1.0], [0.715255, 0.768579, 1.0],
    [0.673683, 0.740423, 1.0], [0.638992, 0.716359, 1.0], [0.609681, 0.695588, 1.0], [0.609681, 0.695588, 1.0], [0.609681, 0.695588, 1.0],
];

/** UsdLuxBlackbodyTemperatureAsRgb: Catmull-Rom over the table, normalized to Rec.709 luminance 1. */
export function usdBlackbodyTemperatureAsRgb(temperature: number): [number, number, number] {
    const x = Math.min(Math.max((temperature - 1000) / 9000, 0), 1) * (kBlackbodyRgb.length - 4);
    const seg = Math.floor(x);
    const t = x - seg;
    const [k0, k1, k2, k3] = [0, 1, 2, 3].map((i) => kBlackbodyRgb[seg + i]!);
    const rgb = [0, 1, 2].map((c) => {
        const [a, b, d, e] = [k0![c]!, k1![c]!, k2![c]!, k3![c]!];
        return 0.5 * (2 * b + (d - a) * t + (2 * a - 5 * b + 4 * d - e) * t * t + (3 * b - a - 3 * d + e) * t * t * t);
    });
    const luma = 0.2126 * rgb[0]! + 0.7152 * rgb[1]! + 0.0722 * rgb[2]!;
    return rgb.map((v) => Math.max(v / luma, 0)) as [number, number, number];
}

function lightIntensity(body: string): float3 {
    const exposure = attrNumber(body, ["inputs:exposure", "exposure"], 0);
    const intensity = attrNumber(body, ["inputs:intensity", "intensity"], 1);
    const color = attrVector(body, ["inputs:color", "color"], [1, 1, 1]);
    let blackbody = [1, 1, 1];
    if (attrNumber(body, ["inputs:enableColorTemperature", "enableColorTemperature"], 0)) {
        blackbody = usdBlackbodyTemperatureAsRgb(attrNumber(body, ["inputs:colorTemperature", "colorTemperature"], 6500));
    }
    const k = Math.pow(2, exposure) * intensity;
    return new float3(k * blackbody[0]! * color[0]!, k * blackbody[1]! * color[1]!, k * blackbody[2]! * color[2]!);
}

/** Cameras, analytic lights and the dome light of a USD layer's text. */
export function extractUsdCamerasAndLights(text: string): { cameras: UsdaCamera[]; lights: AnalyticLight[]; domeLight: UsdaDomeLight | null } {
    const info = usdaStageInfo(text);
    const prims = parseUsdaPrims(text, usdStageRootTransform(info));
    const cameras: UsdaCamera[] = [];
    const lights: AnalyticLight[] = [];
    let domeLight: UsdaDomeLight | null = null;
    const visit = (p: UsdaPrim) => {
        const b = p.body;
        switch (p.type) {
            case "Camera": {
                const focusDistance = Math.max(1, attrNumber(b, ["focusDistance"], 0));
                const view = p.usdWorld;
                const focalLength = attrNumber(b, ["focalLength"], 50);
                const fStop = attrNumber(b, ["fStop"], 0) * (attr(b, ["depthOfField"]) === "false" ? 0 : 1);
                const clip = attrVector(b, ["clippingRange"], [1, 1000000]);
                const cam: UsdaCamera = {
                    name: p.name,
                    position: transformPoint(view, new float3(0, 0, 0)),
                    target: transformPoint(view, new float3(0, 0, -focusDistance)),
                    up: transformVector(view, new float3(0, 1, 0)),
                    focalLength,
                    focalDistance: info.metersPerUnit * focusDistance,
                    apertureRadius: fStop > 0 ? 0.001 * 0.5 * focalLength / fStop : 0,
                    depthRange: [clip[0]! * info.metersPerUnit, clip[1]! * info.metersPerUnit],
                };
                if (attr(b, ["horizontalAperture"]) !== undefined) cam.frameWidth = attrNumber(b, ["horizontalAperture"], 20.955);
                else cam.frameHeight = attrNumber(b, ["verticalAperture"], 15.2908);
                cameras.push(cam);
                break;
            }
            case "DistantLight": {
                const angle = attrNumber(b, ["inputs:angle", "angle"], 0);
                lights.push({ type: LightType.Distant, name: p.name, intensity: lightIntensity(b), dirW: normalize3(transformVector(p.world, new float3(0, 0, -1))), angle: 0.5 * angle * deg });
                break;
            }
            case "SphereLight": {
                const r = attrNumber(b, ["inputs:radius", "radius"], 0.5);
                lights.push({ type: LightType.Sphere, name: p.name, intensity: lightIntensity(b), transMat: mulMat(p.world, matrixFromScaling(new float3(r, r, r))) });
                break;
            }
            case "RectLight": {
                const w = attrNumber(b, ["inputs:width", "width"], 1);
                const h = attrNumber(b, ["inputs:height", "height"], 1);
                lights.push({ type: LightType.Rect, name: p.name, intensity: lightIntensity(b), transMat: mulMat(p.world, matrixFromScaling(new float3(-w / 2, h / 2, -1))) });
                break;
            }
            case "DiskLight": {
                const r = attrNumber(b, ["inputs:radius", "radius"], 0.5);
                lights.push({ type: LightType.Disc, name: p.name, intensity: lightIntensity(b), transMat: mulMat(p.world, matrixFromScaling(new float3(-r, r, -1))) });
                break;
            }
            case "DomeLight": {
                const file = attr(b, ["inputs:texture:file", "texture:file"])?.match(/@([^@]*)@/)?.[1];
                if (!file) {
                    Logger.error(`Failed to resolve environment map path for light '${p.path}'.`);
                    break;
                }
                // USD dome lights are +Z up, Falcor env maps +Y up and offset in longitude.
                const xform = mulMat(mulMat(matrixFromRotationAxisAngle(90 * deg, new float3(0, 1, 0)), p.world), matrixFromRotationAxisAngle(90 * deg, new float3(1, 0, 0)));
                const r = extractEulerAngleXYZ(xform);
                domeLight = {
                    name: p.name,
                    file,
                    intensity: Math.pow(2, attrNumber(b, ["inputs:exposure", "exposure"], 0)) * attrNumber(b, ["inputs:intensity", "intensity"], 1),
                    tint: attrVector(b, ["inputs:color", "color"], [1, 1, 1]) as [number, number, number],
                    rotationDeg: [r.x / deg, r.y / deg, r.z / deg],
                };
                break;
            }
            case "CylinderLight":
            case "GeometryLight":
            case "PortalLight":
                Logger.warning(`USDImporter: unsupported light type '${p.type}' ('${p.path}') ignored.`);
                break;
        }
        p.children.forEach(visit);
    };
    prims.forEach(visit);
    return { cameras, lights, domeLight };
}

/** A UsdPreviewSurface input read from a UsdUVTexture (ConvertedInput::convertTexture). */
export interface UsdaTextureInput {
    /** Path of the UsdUVTexture prim. */
    texture: string;
    /** The texture output the input connects to: r, g, b, a, rg or rgb. */
    output: string;
    /** sourceColorSpace or the file's colorSpace: true sRGB, false raw, undefined per slot. */
    srgb?: boolean;
    /** inputs:scale (native honours it only for emissiveColor, as the emissive factor). */
    scale?: [number, number, number, number];
    /** A UsdTransform2d on st: scale, rotation in degrees, translation. */
    transform?: { scale: [number, number]; rotation: number; translation: [number, number] };
}

/** Texture channel index of a single-channel output name (getChannelIndex), else -1. */
export function usdChannelIndex(output: string): number {
    return ["r", "g", "b", "a"].indexOf(output);
}

/**
 * The textured inputs of every Material's UsdPreviewSurface, keyed by the Material's path, with
 * lowercase input names. Connections are followed through NodeGraph and Material interface
 * attributes to the most upstream source, like native's getSourceInput.
 */
export function extractUsdMaterialTextures(text: string): Map<string, Map<string, UsdaTextureInput>> {
    const byPath = new Map<string, UsdaPrim>();
    const index = (p: UsdaPrim) => {
        byPath.set(p.path, p);
        p.children.forEach(index);
    };
    parseUsdaPrims(text, float4x4.identity()).forEach(index);
    const target = (body: string, name: string) => attr(body, [`${name}.connect`])?.match(/<([^>]+)>/)?.[1];
    // Follows `<prim.attr>` connections to the source prim and its output name.
    const resolve = (connection: string): { prim: UsdaPrim; output: string } | null => {
        for (let hop = 0; hop < 16; hop++) {
            const dot = connection.lastIndexOf(".");
            const prim = byPath.get(connection.slice(0, dot));
            const property = connection.slice(dot + 1);
            if (!prim) return null;
            const next = target(prim.body, property);
            if (!next) return { prim, output: property.replace(/^outputs:/, "") };
            connection = next;
        }
        return null;
    };
    const infoId = (p: UsdaPrim) => attr(p.body, ["info:id"])?.match(/"([^"]*)"/)?.[1];
    const result = new Map<string, Map<string, UsdaTextureInput>>();
    for (const material of byPath.values()) {
        if (material.type !== "Material") continue;
        const surface = target(material.body, "outputs:surface");
        const shader = surface ? resolve(surface)?.prim : undefined;
        if (!shader || infoId(shader) !== "UsdPreviewSurface") continue;
        const inputs = new Map<string, UsdaTextureInput>();
        const names = [...shader.body.matchAll(/^\s*(?:uniform\s+)?[\w\[\]]+\s+inputs:(\w+)\.connect\s*=/gm)].map((m) => m[1]!).sort();
        for (const name of names) {
            const source = resolve(`${shader.path}.inputs:${name}`);
            if (!source || infoId(source.prim) !== "UsdUVTexture") continue;
            const tex = source.prim;
            const input: UsdaTextureInput = { texture: tex.path, output: source.output };
            // Native reads only an un-namespaced sourceColorSpace; the schema's is inputs:sourceColorSpace.
            const colorSpace = attr(tex.body, ["inputs:sourceColorSpace", "sourceColorSpace"])?.match(/"([^"]*)"/)?.[1];
            // The file asset's own colorSpace metadata takes precedence.
            const fileSpace = tex.body.match(/inputs:file\s*=\s*@[^@]*@\s*\(([^)]*)\)/)?.[1]?.match(/colorSpace\s*=\s*"([^"]*)"/)?.[1];
            for (const space of [colorSpace, fileSpace]) {
                if (space === "sRGB") input.srgb = true;
                else if (space === "raw") input.srgb = false;
            }
            const scale = attr(tex.body, ["inputs:scale"]);
            if (scale !== undefined) input.scale = num(scale).slice(0, 4) as [number, number, number, number];
            const st = target(tex.body, "inputs:st");
            const stSource = st ? resolve(st)?.prim : undefined;
            if (stSource && infoId(stSource) === "UsdTransform2d") {
                const s = attrVector(stSource.body, ["inputs:scale"], [1, 1]);
                const t = attrVector(stSource.body, ["inputs:translation"], [0, 0]);
                input.transform = { scale: [s[0]!, s[1]!], rotation: attrNumber(stSource.body, ["inputs:rotation"], 0), translation: [t[0]!, t[1]!] };
            }
            inputs.set(name.toLowerCase(), input);
        }
        result.set(material.path, inputs);
    }
    return result;
}

/**
 * Native's texcoord pre-transform for a material's UsdTransform2d (SceneBuilder applies the
 * inverse of PreviewSurfaceConverter's texture transform): st' = t + R(rotation) * (sx*s, -sy*t).
 * Without a transform it is the plain y-flip (s, -t).
 */
export function usdTexCoordTransform(transform: UsdaTextureInput["transform"]): (s: number, t: number) => [number, number] {
    const { scale = [1, 1], rotation = 0, translation = [0, 0] } = transform ?? {};
    const c = Math.cos(rotation * deg);
    const n = Math.sin(rotation * deg);
    return (s, t) => {
        const x = scale[0] * s;
        const y = -scale[1] * t;
        return [translation[0] + c * x - n * y, translation[1] + n * x + c * y];
    };
}

/** Each prim's bound material path (`material:binding`, inherited by descendants). */
export function extractUsdMaterialBindings(text: string): Map<string, string> {
    const bindings = new Map<string, string>();
    const visit = (p: UsdaPrim, inherited: string | undefined) => {
        const own = p.body.match(/^\s*rel\s+material:binding\s*=\s*<([^>]+)>/m)?.[1] ?? inherited;
        if (own) bindings.set(p.path, own);
        p.children.forEach((c) => visit(c, own));
    };
    parseUsdaPrims(text, float4x4.identity()).forEach((p) => visit(p, undefined));
    return bindings;
}

/** A UsdGeomPointInstancer: its prototypes and per-instance transforms at the earliest time. */
export interface UsdaPointInstancer {
    path: string;
    /** The instancer's local-to-world in USD space (tinyusdz drops its xformOps). */
    usdWorld: float4x4;
    prototypes: string[];
    /** translate * orient * scale (ComputeInstanceTransformsAtTime, ExcludeProtoXform). */
    instances: { proto: number; transform: float4x4 }[];
}

/** The PointInstancers of a USD layer's text (velocities and time samples are not applied). */
export function extractUsdPointInstancers(text: string): UsdaPointInstancer[] {
    const out: UsdaPointInstancer[] = [];
    const visit = (p: UsdaPrim) => {
        if (p.type === "PointInstancer") {
            const b = p.body;
            const rel = b.match(/^\s*rel\s+prototypes\s*=\s*(\[[^\]]*\]|<[^>]+>)/m)?.[1] ?? "";
            const prototypes = [...rel.matchAll(/<([^>]+)>/g)].map((m) => m[1]!);
            const protoIndices = attr(b, ["protoIndices"]);
            if (protoIndices === undefined) {
                Logger.error(`Point instancer '${p.path}' has no prototype indices. Ignoring prim.`);
            } else {
                const indices = num(protoIndices);
                const positions = attrVector(b, ["positions"], []);
                const orientations = attrVector(b, ["orientations"], []);
                const scales = attrVector(b, ["scales"], []);
                if (positions.length !== indices.length * 3) {
                    Logger.error(`Point instancer '${p.path}' has ${indices.length} prototype indices but ${positions.length / 3} transforms.`);
                } else {
                    const instances = indices.map((proto, i) => {
                        let m = matrixFromTranslation(new float3(positions[i * 3]!, positions[i * 3 + 1]!, positions[i * 3 + 2]!));
                        if (orientations.length >= (i + 1) * 4) {
                            // Authored (real, i, j, k); GfRotation normalizes.
                            const [w, x, y, z] = orientations.slice(i * 4, i * 4 + 4) as [number, number, number, number];
                            const len = Math.hypot(w, x, y, z) || 1;
                            m = mulMat(m, matrixFromQuat(new quatf(x / len, y / len, z / len, w / len)));
                        }
                        if (scales.length >= (i + 1) * 3) m = mulMat(m, matrixFromScaling(new float3(scales[i * 3]!, scales[i * 3 + 1]!, scales[i * 3 + 2]!)));
                        return { proto, transform: m };
                    });
                    out.push({ path: p.path, usdWorld: p.usdWorld, prototypes, instances });
                }
            }
        }
        p.children.forEach(visit);
    };
    parseUsdaPrims(text, float4x4.identity()).forEach(visit);
    return out;
}

/** Each prim's first authored display color (`primvars:displayColor`, or the legacy `displayColor`). */
export function extractUsdDisplayColors(text: string): Map<string, [number, number, number]> {
    const colors = new Map<string, [number, number, number]>();
    const visit = (p: UsdaPrim) => {
        const c = num(attr(p.body, ["primvars:displayColor", "displayColor"]) ?? "");
        if (c.length >= 3) colors.set(p.path, [c[0]!, c[1]!, c[2]!]);
        p.children.forEach(visit);
    };
    parseUsdaPrims(text, float4x4.identity()).forEach(visit);
    return colors;
}

/**
 * USDImporter's setMetadata: Omniverse render settings (customLayerData.renderSettings) as
 * Scene::Metadata, with Create's defaults and bounce mapping, plus the subdivision refinement
 * level. Null without a renderSettings dictionary.
 */
export function usdRenderSettings(text: string): { metadata: SceneMetadata; refinementLevel: number } | null {
    const header = text.slice(0, text.search(/^\s*(def|over|class)\s/m) >>> 0);
    const start = header.search(/dictionary\s+renderSettings\s*=\s*\{/);
    if (start < 0) return null;
    let i = header.indexOf("{", start) + 1;
    const from = i;
    for (let depth = 1; i < header.length && depth > 0; i++) depth += header[i] === "{" ? 1 : header[i] === "}" ? -1 : 0;
    const dict = new Map<string, number>();
    for (const m of header.slice(from, i - 1).matchAll(/^\s*\w+\s+"?([\w:]+)"?\s*=\s*([^\n]+)$/gm)) {
        const v = m[2]!.trim();
        const n = /^(true|false)$/.test(v) ? Number(v === "true") : num(v)[0];
        if (n !== undefined) dict.set(m[1]!, n);
    }
    const get = (key: string, fallback: number) => dict.get(key) ?? fallback;
    // Falcor's bounce counts include primary visibility and NEE, Create's do not (subtract 2).
    const diffuse = Math.max(2, Math.floor(get("rtx:pathtracing:maxBounces", 4))) - 2;
    const specTrans = Math.max(2, Math.floor(get("rtx:pathtracing:maxSpecularAndTransmissionBounces", 6))) - 2;
    return {
        metadata: {
            fNumber: get("rtx:post:tonemap:fNumber", 5),
            filmISO: get("rtx:post:tonemap:filmIso", 100),
            shutterSpeed: get("rtx:post:tonemap:cameraShutter", 50),
            samplesPerPixel: Math.floor(get("rtx:pathtracing:spp", 1)),
            maxDiffuseBounces: diffuse,
            maxSpecularBounces: Math.max(diffuse, specTrans),
            maxTransmissionBounces: Math.max(diffuse, specTrans),
            maxVolumeBounces: Math.max(diffuse, Math.max(2, Math.floor(get("rtx:pathtracing:maxVolumeBounces", 4))) - 2),
        },
        refinementLevel: Math.floor(get("rtx:hydra:refinementLevel", 0)),
    };
}

/** An attribute's value text, following bracketed arrays across lines, and its (metadata) block. */
function attrMultiline(body: string, names: string[]): { value: string; metadata: string } | undefined {
    for (const name of names) {
        const escaped = name.replace(/[.:]/g, (c) => `\\${c}`);
        const m = new RegExp(`^\\s*(?:uniform\\s+|custom\\s+)?[\\w\\[\\]]+\\s+${escaped}\\s*=\\s*`, "m").exec(body);
        if (!m) continue;
        const start = m.index + m[0].length;
        const end = body[start] === "[" ? body.indexOf("]", start) + 1 : body.indexOf("\n", start);
        const value = body.slice(start, end > 0 ? end : undefined);
        const metadata = /^\s*\(([^)]*)\)/.exec(body.slice(end > 0 ? end : body.length))?.[1] ?? "";
        return { value, metadata };
    }
    return undefined;
}

/** A UsdGeomMesh's authored topology and subdivision settings (what native's tessellate() reads). */
export interface UsdaSubdivMesh {
    scheme: string;
    orientation: string;
    interpolateBoundary: string;
    faceVaryingLinearInterpolation: string;
    points: number[];
    faceVertexCounts: number[];
    faceVertexIndices: number[];
    /** primvars:st (flattened through its :indices) and its interpolation. */
    st?: { values: number[]; interpolation: string };
    /** The prim's refinementLevel unless refinementEnableOverride is false. */
    refinementLevel?: number;
    skinned: boolean;
    /** normals or primvars:normals is authored (tinyusdz generates smooth ones otherwise). */
    hasNormals: boolean;
    /** Authored normals and their interpolation (normals defaults to vertex). */
    normals?: { values: number[]; interpolation: string };
    holeIndices: number[];
    /** primvars:skel:jointIndices/jointWeights (per point when interpolation is vertex) and skel:joints. */
    skin?: { indices: number[]; weights: number[]; elementSize: number; interpolation: string; joints?: string[] };
    /** Time-sampled points / normals (time codes), for vertex caches. */
    pointsSamples?: { time: number; value: number[] }[];
    normalsSamples?: { time: number; value: number[] }[];
}

/** Every Mesh prim's topology and subdivision settings (USD's default scheme is catmullClark), by path. */
export function extractUsdMeshes(text: string): Map<string, UsdaSubdivMesh> {
    const out = new Map<string, UsdaSubdivMesh>();
    const token = (b: string, name: string, fallback: string) => attr(b, [name])?.match(/"([^"]*)"/)?.[1] ?? fallback;
    const visit = (p: UsdaPrim) => {
        const b = p.body;
        const scheme = token(b, "subdivisionScheme", "catmullClark");
        if (p.type === "Mesh") {
            const mesh: UsdaSubdivMesh = {
                scheme,
                orientation: token(b, "orientation", "rightHanded"),
                interpolateBoundary: token(b, "interpolateBoundary", "edgeAndCorner"),
                faceVaryingLinearInterpolation: token(b, "faceVaryingLinearInterpolation", "cornersPlus1"),
                points: num(attrMultiline(b, ["points"])?.value ?? ""),
                faceVertexCounts: num(attrMultiline(b, ["faceVertexCounts"])?.value ?? ""),
                faceVertexIndices: num(attrMultiline(b, ["faceVertexIndices"])?.value ?? ""),
                skinned: attr(b, ["primvars:skel:jointIndices"]) !== undefined && attr(b, ["primvars:skel:jointWeights"]) !== undefined,
                hasNormals: attrMultiline(b, ["normals", "primvars:normals"]) !== undefined || attrTimeSamples(b, "normals") !== undefined,
                holeIndices: num(attrMultiline(b, ["holeIndices"])?.value ?? ""),
                pointsSamples: attrTimeSamples(b, "points"),
                normalsSamples: attrTimeSamples(b, "normals"),
            };
            // getTexCoordPrimvar: primvars:st, primvars:st_0, else a texCoord2-typed primvar.
            const typed = b.match(/^\s*texCoord2[fdh]\[\]\s+(primvars:[\w:]+?)\s*=/m)?.[1];
            for (const name of ["primvars:st", "primvars:st_0", ...(typed ? [typed] : [])]) {
                const values = attrMultiline(b, [name]);
                if (values === undefined) continue;
                let st = num(values.value);
                const indices = attrMultiline(b, [`${name}:indices`]);
                if (indices !== undefined) st = num(indices.value).flatMap((i) => [st[i * 2]!, st[i * 2 + 1]!]);
                mesh.st = { values: st, interpolation: values.metadata.match(/interpolation\s*=\s*"(\w+)"/)?.[1] ?? "constant" };
                break;
            }
            const normals = attrMultiline(b, ["primvars:normals"]) ?? attrMultiline(b, ["normals"]);
            if (normals) {
                const fallback = attrMultiline(b, ["primvars:normals"]) ? "constant" : "vertex";
                mesh.normals = { values: num(normals.value), interpolation: normals.metadata.match(/interpolation\s*=\s*"(\w+)"/)?.[1] ?? fallback };
            } else if (mesh.normalsSamples) {
                const meta = b.match(/normals\.timeSamples[^]*?\}\s*\(([^)]*)\)/)?.[1] ?? "";
                mesh.normals = { values: mesh.normalsSamples[0]!.value, interpolation: meta.match(/interpolation\s*=\s*"(\w+)"/)?.[1] ?? "vertex" };
            }
            const ji = attrMultiline(b, ["primvars:skel:jointIndices"]);
            const jw = attrMultiline(b, ["primvars:skel:jointWeights"]);
            if (ji && jw) {
                const meta = (m: string, key: string) => m.match(new RegExp(`${key}\\s*=\\s*"?(\\w+)"?`))?.[1];
                mesh.skin = {
                    indices: num(ji.value),
                    weights: num(jw.value),
                    elementSize: Number(meta(ji.metadata, "elementSize") ?? 1),
                    interpolation: meta(ji.metadata, "interpolation") ?? "constant",
                    joints: attrMultiline(b, ["skel:joints"])?.value.match(/"([^"]*)"/g)?.map((t) => t.slice(1, -1)),
                };
            }
            if (attr(b, ["refinementEnableOverride"]) !== "false" && attr(b, ["refinementLevel"]) !== undefined) mesh.refinementLevel = attrNumber(b, ["refinementLevel"], 0);
            out.set(p.path, mesh);
        }
        p.children.forEach(visit);
    };
    parseUsdaPrims(text, float4x4.identity()).forEach(visit);
    return out;
}

/** An attribute's time samples (`name.timeSamples = { time: value, ... }`), ascending, or undefined. */
function attrTimeSamples(body: string, name: string): { time: number; value: number[] }[] | undefined {
    const escaped = name.replace(/[.:]/g, (c) => `\\${c}`);
    const m = new RegExp(`^\\s*(?:uniform\\s+|custom\\s+)?[\\w\\[\\]]+\\s+${escaped}\\.timeSamples\\s*=\\s*\\{`, "m").exec(body);
    if (!m) return undefined;
    let i = m.index + m[0].length;
    const from = i;
    for (let depth = 1; i < body.length && depth > 0; i++) depth += body[i] === "{" ? 1 : body[i] === "}" ? -1 : 0;
    const samples = [...body.slice(from, i - 1).matchAll(/([-+]?[\d.]+(?:[eE][-+]?\d+)?)\s*:\s*(\[[^\]]*\]|\([^)]*\)|[-+\w.]+)/g)].map((s) => ({ time: Number(s[1]), value: num(s[2]!) }));
    return samples.sort((a, b) => a.time - b.time);
}

/** USD's linear interpolation of time samples (held outside the range). */
export function sampleAt(samples: { time: number; value: number[] }[], t: number): number[] {
    if (t <= samples[0]!.time) return samples[0]!.value;
    const last = samples[samples.length - 1]!;
    if (t >= last.time) return last.value;
    const k = samples.findIndex((s) => s.time > t);
    const [a, b] = [samples[k - 1]!, samples[k]!];
    const u = (t - a.time) / (b.time - a.time);
    return a.value.map((v, c) => v + (b.value[c]! - v) * u);
}

/**
 * quatFromAngleAxis(radians(degrees)) in float precision, as native builds keyframes: cos(90°) is
 * then -4.4e-8, not +6e-17, which decides slerp's direction for 180° turns.
 */
function quatFromAngleAxisF32(degrees: number, axis: float3): quatf {
    const f = Math.fround;
    const half = f(f(f(degrees) * f(Math.PI / 180)) * 0.5);
    const [sn, cs] = [f(Math.sin(half)), f(Math.cos(half))];
    return new quatf(axis.x * sn, axis.y * sn, axis.z * sn, cs);
}

/** An animated xformable's keyframes (ImporterContext::createKeyframe), times in seconds. */
export interface UsdaXformAnimation {
    times: number[];
    translation: float3[];
    rotation: quatf[];
    scaling: float3[];
}

/** Stage timeCodesPerSecond (else framesPerSecond, else 24, as UsdStage::GetTimeCodesPerSecond). */
export function usdTimeCodesPerSecond(text: string): number {
    const header = text.slice(0, text.search(/^\s*(def|over|class)\s/m) >>> 0);
    const get = (key: string) => num(header.match(new RegExp(`\\b${key}\\s*=\\s*([-+\\d.eE]+)`))?.[1] ?? "")[0];
    return get("timeCodesPerSecond") ?? get("framesPerSecond") ?? 24;
}

/**
 * Prims whose xformOps are time-sampled, keyed by path: keyframes at the union of the ops'
 * sample times, from the XformCommonAPI vectors (translate, one rotate op with its order,
 * scale; pivots ignored with a warning, as natively).
 */
export function extractUsdXformAnimations(text: string): Map<string, UsdaXformAnimation> {
    const tcps = usdTimeCodesPerSecond(text);
    const out = new Map<string, UsdaXformAnimation>();
    const axes = { X: new float3(1, 0, 0), Y: new float3(0, 1, 0), Z: new float3(0, 0, 1) } as const;
    const visit = (p: UsdaPrim) => {
        const b = p.body;
        const ops = attr(b, ["xformOpOrder"])?.match(/"([^"]+)"/g)?.map((s) => s.slice(1, -1)) ?? [];
        const sampled = ops.map((op) => attrTimeSamples(b, op.replace(/^!invert!/, "")));
        if (sampled.some((s) => s && s.length > 0)) {
            const times = [...new Set(sampled.flatMap((s) => s?.map((x) => x.time) ?? []))].sort((x, y) => x - y);
            const value = (op: string, i: number, t: number) => {
                const s = sampled[i];
                return s && s.length > 0 ? sampleAt(s, t) : num(attr(b, [op]) ?? "");
            };
            const anim: UsdaXformAnimation = { times: times.map((t) => t / tcps), translation: [], rotation: [], scaling: [] };
            let warnedPivot = false;
            for (const t of times) {
                let tr = new float3(0, 0, 0);
                let sc = new float3(1, 1, 1);
                let rot = new quatf(0, 0, 0, 1);
                ops.forEach((op, i) => {
                    if (op.startsWith("!invert!")) return;
                    const kind = op.split(":")[1] ?? "";
                    const v = value(op, i, t);
                    if (kind === "translate" && op.split(":")[2] === "pivot") {
                        if (!warnedPivot && v.some((x) => x !== 0)) Logger.warning(`Ignoring non-zero pivot extracted from '${p.path}'.`);
                        warnedPivot = true;
                    } else if (kind === "translate") tr = new float3(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0);
                    else if (kind === "scale") sc = new float3(v[0] ?? 1, v[1] ?? 1, v[2] ?? 1);
                    else if (/^rotate[XYZ]{1,3}$/.test(kind)) {
                        const letters = kind.slice(6).split("") as ("X" | "Y" | "Z")[];
                        const degrees = (l: "X" | "Y" | "Z") => (letters.length === 1 ? (v[0] ?? 0) : (v["XYZ".indexOf(l)] ?? 0));
                        // The first axis applies first: q = q_last * ... * q_first (native's order table).
                        rot = letters.reduce((q, l) => mulQuat(quatFromAngleAxisF32(degrees(l), axes[l]), q), new quatf(0, 0, 0, 1));
                    } else {
                        Logger.warning(`USDImporter: time-sampled xformOp '${op}' on '${p.path}' is not an XformCommonAPI op; ignored.`);
                    }
                });
                anim.translation.push(tr);
                anim.rotation.push(rot);
                anim.scaling.push(sc);
            }
            out.set(p.path, anim);
        }
        p.children.forEach(visit);
    };
    parseUsdaPrims(text, float4x4.identity()).forEach(visit);
    return out;
}

/** A UsdSkelSkeleton with its bound SkelAnimation (ImporterContext::createSkeleton). */
export interface UsdaSkeleton {
    path: string;
    /** Joint paths ("Root", "Root/Arm"), in skeleton order. */
    joints: string[];
    /** Skeleton-space bind transforms. */
    bind: float4x4[];
    /** Joint-local rest transforms. */
    rest: float4x4[];
    /** Per-sample joint-local components for every skeleton joint (rest for joints the animation lacks). */
    anim: { times: number[]; translation: float3[][]; rotation: quatf[][]; scale: float3[][] } | null;
}

/** USD matrix4d[] text (row-major, row vectors) as column-vector float4x4s. */
function usdMatrices(text: string): float4x4[] {
    const v = num(text);
    return Array.from({ length: Math.floor(v.length / 16) }, (_, i) => float4x4.fromRows([0, 1, 2, 3].map((r) => [0, 1, 2, 3].map((c) => v[i * 16 + c * 4 + r]!))));
}

/** Skeletons by path, and each prim's bound skeleton (`skel:skeleton`, inherited by descendants). */
export function extractUsdSkeletons(text: string): { skeletons: Map<string, UsdaSkeleton>; bindings: Map<string, string> } {
    const byPath = new Map<string, UsdaPrim>();
    const bindings = new Map<string, string>();
    const index = (p: UsdaPrim, inherited: string | undefined) => {
        byPath.set(p.path, p);
        const own = p.body.match(/^\s*rel\s+skel:skeleton\s*=\s*<([^>]+)>/m)?.[1] ?? inherited;
        if (own) bindings.set(p.path, own);
        p.children.forEach((c) => index(c, own));
    };
    parseUsdaPrims(text, float4x4.identity()).forEach((p) => index(p, undefined));
    const tokens = (b: string, name: string) => attrMultiline(b, [name])?.value.match(/"([^"]*)"/g)?.map((t) => t.slice(1, -1)) ?? [];
    const skeletons = new Map<string, UsdaSkeleton>();
    for (const p of byPath.values()) {
        if (p.type !== "Skeleton") continue;
        const joints = tokens(p.body, "joints");
        const bind = usdMatrices(attrMultiline(p.body, ["bindTransforms"])?.value ?? "");
        const parentOf = (i: number) => joints.indexOf(joints[i]!.slice(0, Math.max(0, joints[i]!.lastIndexOf("/"))));
        let rest = usdMatrices(attrMultiline(p.body, ["restTransforms"])?.value ?? "");
        if (rest.length !== joints.length) {
            // Without rest transforms, the bind pose in joint-local form.
            rest = joints.map((_, i) => (parentOf(i) >= 0 ? mulMat(inverse(bind[parentOf(i)]!), bind[i]!) : bind[i] ?? float4x4.identity()));
        }
        const skel: UsdaSkeleton = { path: p.path, joints, bind, rest, anim: null };
        const animPrim = byPath.get(p.body.match(/^\s*rel\s+skel:animationSource\s*=\s*<([^>]+)>/m)?.[1] ?? "");
        if (animPrim && animPrim.type === "SkelAnimation") {
            const b = animPrim.body;
            const animJoints = tokens(b, "joints");
            const channel = (name: string, width: number) => {
                const samples = attrTimeSamples(b, name);
                const fallback = attrMultiline(b, [name]);
                return { samples, value: fallback ? num(fallback.value) : undefined, width };
            };
            const [tr, rot, sc] = [channel("translations", 3), channel("rotations", 4), channel("scales", 3)];
            const times = [...new Set([tr, rot, sc].flatMap((c) => c.samples?.map((s) => s.time) ?? []))].sort((x, y) => x - y);
            const at = (c: ReturnType<typeof channel>, t: number): number[] | undefined => {
                if (!c.samples || c.samples.length === 0) return c.value;
                if (c.width !== 4) return sampleAt(c.samples, t);
                // Quaternions interpolate by slerp (USD's GfSlerp), per joint.
                const k = c.samples.findIndex((s) => s.time >= t);
                if (k <= 0) return c.samples[k < 0 ? c.samples.length - 1 : 0]!.value;
                const [a, bb] = [c.samples[k - 1]!, c.samples[k]!];
                const u = (t - a.time) / (bb.time - a.time);
                return a.value.flatMap((_, i) => {
                    if (i % 4) return [];
                    const q = slerp(new quatf(a.value[i + 1]!, a.value[i + 2]!, a.value[i + 3]!, a.value[i]!), new quatf(bb.value[i + 1]!, bb.value[i + 2]!, bb.value[i + 3]!, bb.value[i]!), u);
                    return [q.w, q.x, q.y, q.z];
                });
            };
            if (times.length > 0) {
                skel.anim = { times, translation: [], rotation: [], scale: [] };
                for (const t of times) {
                    const [tv, rv, sv] = [at(tr, t), at(rot, t), at(sc, t)];
                    const pose = joints.map((name, i) => {
                        const restTRS = decomposeTRS(rest[i]!);
                        const a = animJoints.indexOf(name);
                        if (a < 0) return restTRS;
                        return {
                            t: tv ? new float3(tv[a * 3]!, tv[a * 3 + 1]!, tv[a * 3 + 2]!) : restTRS.t,
                            r: rv ? new quatf(rv[a * 4 + 1]!, rv[a * 4 + 2]!, rv[a * 4 + 3]!, rv[a * 4]!) : restTRS.r,
                            s: sv ? new float3(sv[a * 3]!, sv[a * 3 + 1]!, sv[a * 3 + 2]!) : restTRS.s,
                        };
                    });
                    skel.anim.translation.push(pose.map((j) => j.t));
                    skel.anim.rotation.push(pose.map((j) => j.r));
                    skel.anim.scale.push(pose.map((j) => j.s));
                }
            }
        }
        skeletons.set(p.path, skel);
    }
    return { skeletons, bindings };
}
