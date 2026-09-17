/**
 * Mitsuba 3 scene importer (Scene/Importer) — mirrors upstream Falcor's
 * plugins/importers/MitsubaImporter, which itself supports a subset of the
 * Mitsuba XML format. A `.xml` scene is a *whole scene* (sensor + emitters +
 * shapes), so like the pbrt importer this drives a full SceneBuilder.
 *
 * Coverage, matching the upstream plugin exactly:
 *   - Sensor: perspective, thinlens (fov/focal_length, near/far clip, focus
 *     distance, aperture radius; the film's resolution feeds the fov mapping)
 *   - Shapes: obj, ply, sphere, disk, rectangle, cube (+ to_world)
 *   - BSDFs: diffuse, conductor, roughconductor, dielectric, roughdielectric,
 *     thindielectric, plastic, roughplastic, twosided
 *   - Emitters: area (nested in a shape), constant, envmap
 *   - Textures: bitmap, checkerboard
 *   - Media: homogeneous (interior) -> volume scattering/absorption
 *   - Transforms: translate, rotate, scale, lookat, matrix
 *
 * Upstream's parser rejects `<default>`, `<include>`, `<alias>` and `<path>`
 * ("not implemented"), and reads `<spectrum>` as if it were `<rgb>`; both are
 * mirrored, so scenes those features appear in fail here exactly as they do
 * natively. `point` emitters are likewise dropped with a warning, as upstream
 * does. Documented divergences (docs §9):
 *   - `twosided` dereferences a null material upstream (it crashes); here the
 *     inner material is marked double-sided, which is the evident intent.
 *   - Per-texture `to_uv` transforms have no web equivalent (one uv set per
 *     primitive) and warn instead of applying.
 *   - The procedural `checkerboard` texture is RGBA8 rather than RGBA32Float,
 *     since the port packs material textures into one RGBA8 array (§6.2).
 */

import type { Device } from "../../Core/API/Device.js";
import type { Scene } from "../Scene.js";
import { MaterialType } from "../Material/MaterialData.js";
import { float2, float3, float4, normalize3, sub3, cross, length3 } from "../../Utils/Math/Vector.js";
import { float4x4, mulMat, matrixFromTranslation, matrixFromScaling, matrixFromRotationAxisAngle, transformPoint, transformVector } from "../../Utils/Math/Matrix.js";
import { fovYToFocalLength } from "../Camera/Camera.js";
import { RuntimeError } from "../../Core/Error.js";
import { withScriptSearchPath } from "../../Core/AssetResolver.js";
import { Logger } from "../../Utils/Logger.js";
import { SceneBuilderBridge, CameraBridge, MaterialBridge, TriangleMesh, type TriangleMeshDesc } from "../SceneBuilder.js";

const radians = (deg: number) => (deg * Math.PI) / 180;
const degrees = (rad: number) => (rad * 180) / Math.PI;

/** Mitsuba's named IOR table (Tables.h). */
const kIORTable: Record<string, number> = {
    vacuum: 1.0,
    helium: 1.000036,
    hydrogen: 1.000132,
    air: 1.000277,
    "carbon dioxide": 1.00045,
    water: 1.333,
    acetone: 1.36,
    ethanol: 1.361,
    "carbon tetrachloride": 1.461,
    glycerol: 1.4729,
    benzene: 1.501,
    "silicone oil": 1.52045,
    bromine: 1.661,
    "water ice": 1.31,
    "fused quartz": 1.458,
    pyrex: 1.47,
    "acrylic glass": 1.49,
    polypropylene: 1.49,
    bk7: 1.5046,
    "sodium chloride": 1.544,
    amber: 1.55,
    pet: 1.575,
    diamond: 2.419,
};

/** Tag kinds (Parser.h kTags). Object tags share `Tag.Object`. */
type Tag =
    | "boolean" | "integer" | "float" | "string" | "point" | "vector" | "spectrum" | "rgb"
    | "transform" | "translate" | "matrix" | "rotate" | "scale" | "lookat"
    | "object" | "ref" | "include" | "alias" | "default" | "path";

/** Mitsuba plugin classes (Parser.h kClasses). */
type Class = "scene" | "integrator" | "sensor" | "emitter" | "sampler" | "film" | "rfilter" | "shape" | "bsdf" | "texture" | "medium";

const kClassNames: Class[] = ["scene", "integrator", "sensor", "emitter", "sampler", "film", "rfilter", "shape", "bsdf", "texture", "medium"];

const kTags: Record<string, Tag> = {
    boolean: "boolean",
    integer: "integer",
    float: "float",
    string: "string",
    point: "point",
    vector: "vector",
    transform: "transform",
    translate: "translate",
    matrix: "matrix",
    rotate: "rotate",
    scale: "scale",
    lookat: "lookat",
    ref: "ref",
    spectrum: "spectrum",
    rgb: "rgb",
    include: "include",
    alias: "alias",
    default: "default",
    path: "path",
    ...Object.fromEntries(kClassNames.map((c) => [c, "object" as Tag])),
};

type PropValue = boolean | number | string | float3 | float4x4;

/** Mirrors Mitsuba::Properties: typed values plus named references. */
class Properties {
    private values = new Map<string, { type: string; value: PropValue }>();
    /** Named references; native holds a std::map, so iteration is sorted by name. */
    private refs = new Map<string, string>();

    set(type: string, name: string, value: PropValue): void {
        this.values.set(name, { type, value });
    }
    addRef(name: string, id: string): void {
        if (!this.refs.has(name)) this.refs.set(name, id); // std::map::emplace keeps the first
    }
    /** Named references in name order, as native's std::map iterates them. */
    namedReferences(): [string, string][] {
        return [...this.refs.entries()].sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    }
    hasRef(name: string): boolean {
        return this.refs.has(name);
    }
    getRef(name: string): string {
        return this.refs.get(name)!;
    }
    has(type: string, name: string): boolean {
        return this.values.get(name)?.type === type;
    }
    private get<T extends PropValue>(type: string, name: string, def?: T): T {
        const entry = this.values.get(name);
        if (!entry || entry.type !== type) {
            if (def === undefined) throw new RuntimeError(`Mitsuba: missing ${type} property '${name}'`);
            return def;
        }
        return entry.value as T;
    }
    getBool(name: string, def?: boolean): boolean {
        return this.get("boolean", name, def);
    }
    getInt(name: string, def?: number): number {
        return this.get("integer", name, def);
    }
    getFloat(name: string, def?: number): number {
        return this.get("float", name, def);
    }
    getString(name: string, def?: string): string {
        return this.get("string", name, def);
    }
    getFloat3(name: string, def?: float3): float3 {
        return this.get("float3", name, def);
    }
    getColor3(name: string, def?: float3): float3 {
        return this.get("color3", name, def);
    }
    getTransform(name: string, def?: float4x4): float4x4 {
        return this.get("transform", name, def);
    }
}

interface XMLObject {
    id: string;
    cls: Class;
    type: string;
    props: Properties;
}

// -------------------------------------------------------------------------
// Parsing (mirrors Parser.h)
// -------------------------------------------------------------------------

function splitList(value: string, sep: string): string[] {
    return value.split(sep).map((s) => s.trim());
}

/** Mirrors Mitsuba::parseFloat: the whole token must be a number. */
function parseFloatStrict(s: string, what: string): number {
    const value = Number(s.trim());
    if (s.trim() === "" || !Number.isFinite(value)) throw new RuntimeError(`Mitsuba: could not parse ${what} '${s}'`);
    return value;
}

/** Mirrors checkAttributes: unexpected attributes and (optionally) missing ones are errors. */
function checkAttributes(node: Element, expected: string[], expectAll = true): void {
    const remaining = new Set(expected);
    let foundOne = false;
    for (const attr of Array.from(node.attributes)) {
        if (!remaining.has(attr.name)) throw new RuntimeError(`Mitsuba: unexpected attribute '${attr.name}' in node '${node.nodeName}'`);
        remaining.delete(attr.name);
        foundOne = true;
    }
    if (remaining.size > 0 && (!foundOne || expectAll)) {
        throw new RuntimeError(`Mitsuba: missing attribute '${[...remaining][0]}' in node '${node.nodeName}'`);
    }
}

/** Mirrors expandValueToXYZ: 'value' splits into x/y/z (1 or 3 components). */
function expandValueToXYZ(node: Element): void {
    const value = node.getAttribute("value");
    if (value === null) return;
    const tokens = splitList(value, ",");
    if (node.hasAttribute("x") || node.hasAttribute("y") || node.hasAttribute("z")) {
        throw new RuntimeError("Mitsuba: can't mix and match 'value' and 'x'/'y'/'z' attributes");
    }
    if (tokens.length === 1) {
        for (const axis of "xyz") node.setAttribute(axis, tokens[0]!);
    } else if (tokens.length === 3) {
        "xyz".split("").forEach((axis, i) => node.setAttribute(axis, tokens[i]!));
    } else {
        throw new RuntimeError("Mitsuba: 'value' attribute must have exactly 1 or 3 elements");
    }
    node.removeAttribute("value");
}

/** Mirrors parseVector: missing components take `defaultValue`. */
function parseVector(node: Element, defaultValue = 0): float3 {
    const comp = (axis: string) => {
        const value = node.getAttribute(axis);
        return value === null || value === "" ? defaultValue : parseFloatStrict(value, "floating point value");
    };
    return new float3(comp("x"), comp("y"), comp("z"));
}

/** Mirrors parseNamedVector: a 3-component attribute like origin="1, 2, 3". */
function parseNamedVector(node: Element, name: string): float3 {
    const list = splitList(node.getAttribute(name) ?? "", ",");
    if (list.length !== 3) throw new RuntimeError(`Mitsuba: '${name}' attribute must have exactly 3 elements`);
    return new float3(...(list.map((v) => parseFloatStrict(v, "floating point value")) as [number, number, number]));
}

/** Mirrors parseRGB: one or three components. */
function parseRGB(node: Element): float3 {
    const value = node.getAttribute("value") ?? "";
    let tokens = splitList(value, ",");
    if (tokens.length === 1) tokens = [tokens[0]!, tokens[0]!, tokens[0]!];
    if (tokens.length !== 3) throw new RuntimeError(`Mitsuba: RGB value requires one or three components (got '${value}')`);
    return new float3(...(tokens.map((t) => parseFloatStrict(t, "RGB value")) as [number, number, number]));
}

/** Mirrors Mitsuba::Version comparison against 2.0.0. */
function isLegacyVersion(version: string): boolean {
    const [major = 0, minor = 0, patch = 0] = version.split(".").map((v) => parseInt(v, 10) || 0);
    return major < 2 || (major === 2 && (minor < 0 || (minor === 0 && patch < 0)));
}

/** Mirrors upgradeTree: brings pre-2.0.0 scenes up to the modern spelling. */
function upgradeTree(root: Element): void {
    const all: Element[] = [];
    const collect = (node: Element) => {
        all.push(node);
        for (const child of Array.from(node.children)) collect(child);
    };
    collect(root);

    for (const node of all) {
        // camelCase -> underscore_case on 'name' attributes.
        if (node.hasAttribute("name") && node.nodeName !== "default") {
            let name = node.getAttribute("name")!;
            for (let i = 0; i + 1 < name.length; i++) {
                const a = name[i]!;
                const b = name[i + 1]!;
                if (a >= "a" && a <= "z" && b >= "A" && b <= "Z") {
                    name = name.slice(0, i + 1) + "_" + name.slice(i + 1);
                    i += 2;
                    while (i < name.length && name[i]! >= "A" && name[i]! <= "Z") {
                        name = name.slice(0, i) + name[i]!.toLowerCase() + name.slice(i + 1);
                        i++;
                    }
                }
            }
            node.setAttribute("name", name);
        }
        // Reserved identifiers.
        const id = node.getAttribute("id");
        if (id && id.startsWith("_")) node.setAttribute("id", `ID${id}__UPGR`);
    }

    // <lookAt> -> <lookat>; the DOM cannot rename in place, so rebuild the node.
    for (const node of all) {
        if (node.nodeName !== "lookAt") continue;
        const renamed = node.ownerDocument.createElement("lookat");
        for (const attr of Array.from(node.attributes)) renamed.setAttribute(attr.name, attr.value);
        node.parentNode?.replaceChild(renamed, node);
    }

    // diffuse BSDFs: diffuse_reflectance -> reflectance.
    for (const node of all) {
        if (node.nodeName !== "bsdf" || node.getAttribute("type") !== "diffuse") continue;
        for (const child of Array.from(node.children)) {
            if (child.getAttribute("name") === "diffuse_reflectance") child.setAttribute("name", "reflectance");
        }
    }

    // uoffset/voffset/uscale/vscale -> a to_uv transform block.
    for (const node of all) {
        const named = (name: string) => Array.from(node.children).find((c) => c.nodeName === "float" && c.getAttribute("name") === name);
        const parts = { uoffset: named("uoffset"), voffset: named("voffset"), uscale: named("uscale"), vscale: named("vscale") };
        if (!parts.uoffset && !parts.voffset && !parts.uscale && !parts.vscale) continue;
        const offset = new float2(0, 0);
        const scale = new float2(1, 1);
        const take = (element: Element | undefined): number => {
            const value = parseFloatStrict(element!.getAttribute("value") ?? "", "floating point value");
            node.removeChild(element!);
            return value;
        };
        if (parts.uoffset) offset.x = take(parts.uoffset);
        if (parts.voffset) offset.y = take(parts.voffset);
        if (parts.uscale) scale.x = take(parts.uscale);
        if (parts.vscale) scale.y = take(parts.vscale);
        const transform = node.ownerDocument.createElement("transform");
        transform.setAttribute("name", "to_uv");
        if (offset.x !== 0 || offset.y !== 0) {
            const element = node.ownerDocument.createElement("translate");
            element.setAttribute("x", String(offset.x));
            element.setAttribute("y", String(offset.y));
            transform.appendChild(element);
        }
        if (scale.x !== 1 || scale.y !== 1) {
            const element = node.ownerDocument.createElement("scale");
            element.setAttribute("x", String(scale.x));
            element.setAttribute("y", String(scale.y));
            transform.appendChild(element);
        }
        node.appendChild(transform);
    }
}

/** Parser state shared across the tree (mirrors Mitsuba::XMLContext). */
interface ParseContext {
    instances: Map<string, XMLObject>;
    idCounter: number;
    /** The transform under construction; reset by each <transform> node. */
    transform: float4x4;
}

/**
 * Mirrors Mitsuba::parseXML. Returns the (name, id) pair a parent uses to record
 * a named reference, or empty strings for plain properties.
 */
function parseXML(ctx: ParseContext, node: Element, parentTag: Tag | null, props: Properties, counter: { arg: number }, depth = 0): [string, string] {
    const tag = kTags[node.nodeName];
    if (tag === undefined) throw new RuntimeError(`Mitsuba: unexpected tag '${node.nodeName}'`);

    const hasParent = parentTag !== null;
    const parentIsObject = hasParent && parentTag === "object";
    const currentIsObject = tag === "object";
    const parentIsTransform = parentTag === "transform";
    const currentIsTransformOp = tag === "translate" || tag === "rotate" || tag === "scale" || tag === "lookat" || tag === "matrix";

    if (!hasParent && !currentIsObject) throw new RuntimeError(`Mitsuba: root node '${node.nodeName}' must be an object`);
    if (parentIsTransform !== currentIsTransformOp) {
        throw new RuntimeError(
            parentIsTransform ? "Mitsuba: transform nodes can only contain transform operations" : "Mitsuba: transform operations can only occur in a transform node",
        );
    }
    if (hasParent && !parentIsObject && !(parentIsTransform && currentIsTransformOp)) {
        throw new RuntimeError(`Mitsuba: node '${node.nodeName}' cannot occur as child of a property`);
    }

    if (depth === 0) {
        const version = node.getAttribute("version");
        if (version === null) throw new RuntimeError(`Mitsuba: missing version attribute in root element '${node.nodeName}'`);
        if (isLegacyVersion(version)) upgradeTree(node);
        node.removeAttribute("version");
    }

    if (node.nodeName === "scene") node.setAttribute("type", "scene");

    // Names and ids: generated ones keep declaration order (zero-padded so the
    // sorted reference map iterates in file order).
    if (node.hasAttribute("name")) {
        const name = node.getAttribute("name")!;
        if (name === "") throw new RuntimeError(`Mitsuba: node '${node.nodeName}' has empty name attribute`);
        if (name.startsWith("_")) throw new RuntimeError(`Mitsuba: node '${node.nodeName}' has invalid name '${name}' with leading underscores`);
    } else if (currentIsObject || tag === "ref") {
        node.setAttribute("name", `_arg_${String(counter.arg++).padStart(4, "0")}`);
    }
    if (node.hasAttribute("id")) {
        const id = node.getAttribute("id")!;
        if (id === "") throw new RuntimeError(`Mitsuba: node '${node.nodeName}' has empty id attribute`);
        if (id.startsWith("_")) throw new RuntimeError(`Mitsuba: node '${node.nodeName}' has invalid id '${id}' with leading underscores`);
    } else if (currentIsObject) {
        node.setAttribute("id", `_unnamed_${ctx.idCounter++}`);
    }

    const name = node.getAttribute("name") ?? "";

    switch (tag) {
        case "object": {
            checkAttributes(node, ["type", "id", "name"]);
            const id = node.getAttribute("id")!;
            const type = node.getAttribute("type") ?? "";
            if (ctx.instances.has(id)) throw new RuntimeError(`Mitsuba: node '${node.nodeName}' has duplicate id '${id}'`);

            const nested = new Properties();
            for (const child of Array.from(node.children)) {
                const [nestedName, nestedID] = parseXML(ctx, child, tag, nested, counter, depth + 1);
                if (nestedID !== "") nested.addRef(nestedName, nestedID);
            }
            ctx.instances.set(id, { id, cls: node.nodeName as Class, type, props: nested });
            return [name, id];
        }
        case "ref": {
            checkAttributes(node, ["name", "id"]);
            return [name, node.getAttribute("id") ?? ""];
        }
        case "alias":
        case "default":
        case "path":
        case "include":
            // Upstream's parser throws "not implemented" for all four.
            throw new RuntimeError(`Mitsuba: <${node.nodeName}> is not implemented`);
        case "string": {
            checkAttributes(node, ["name", "value"]);
            props.set("string", name, node.getAttribute("value") ?? "");
            break;
        }
        case "float": {
            checkAttributes(node, ["name", "value"]);
            props.set("float", name, parseFloatStrict(node.getAttribute("value") ?? "", "floating point value"));
            break;
        }
        case "integer": {
            checkAttributes(node, ["name", "value"]);
            const value = (node.getAttribute("value") ?? "").trim();
            if (!/^[+-]?\d+$/.test(value)) throw new RuntimeError(`Mitsuba: could not parse integer value '${value}'`);
            props.set("integer", name, parseInt(value, 10));
            break;
        }
        case "boolean": {
            checkAttributes(node, ["name", "value"]);
            const value = (node.getAttribute("value") ?? "").toLowerCase();
            if (value !== "true" && value !== "false") throw new RuntimeError(`Mitsuba: could not parse boolean value '${value}', must be 'true' or 'false'`);
            props.set("boolean", name, value === "true");
            break;
        }
        case "vector":
        case "point": {
            expandValueToXYZ(node);
            checkAttributes(node, ["name", "x", "y", "z"]);
            props.set("float3", name, parseVector(node));
            break;
        }
        case "rgb":
        case "spectrum": {
            // Upstream reads <spectrum> with the RGB parser; wavelength lists fail.
            checkAttributes(node, ["name", "value"]);
            props.set("color3", name, parseRGB(node));
            break;
        }
        case "transform": {
            checkAttributes(node, ["name"]);
            ctx.transform = float4x4.identity();
            break;
        }
        case "rotate": {
            expandValueToXYZ(node);
            checkAttributes(node, ["angle", "x", "y", "z"], false);
            const axis = parseVector(node);
            const angle = radians(parseFloatStrict(node.getAttribute("angle") ?? "", "floating point value"));
            ctx.transform = mulMat(matrixFromRotationAxisAngle(angle, axis), ctx.transform);
            break;
        }
        case "translate": {
            expandValueToXYZ(node);
            checkAttributes(node, ["x", "y", "z"], false);
            ctx.transform = mulMat(matrixFromTranslation(parseVector(node)), ctx.transform);
            break;
        }
        case "scale": {
            expandValueToXYZ(node);
            checkAttributes(node, ["x", "y", "z"], false);
            ctx.transform = mulMat(matrixFromScaling(parseVector(node, 1)), ctx.transform);
            break;
        }
        case "lookat": {
            if (!node.hasAttribute("up")) node.setAttribute("up", "0,0,0");
            checkAttributes(node, ["origin", "target", "up"]);
            const origin = parseNamedVector(node, "origin");
            const target = parseNamedVector(node, "target");
            let up = parseNamedVector(node, "up");
            const dir = normalize3(sub3(target, origin));
            if (length3(up) === 0) up = buildFrameTangent(dir);
            const right = normalize3(cross(normalize3(up), dir));
            const newUp = cross(dir, right);
            const m = float4x4.identity();
            const setCol = (c: number, v: float3, w: number) => {
                m.set(0, c, v.x);
                m.set(1, c, v.y);
                m.set(2, c, v.z);
                m.set(3, c, w);
            };
            setCol(0, right, 0);
            setCol(1, newUp, 0);
            setCol(2, dir, 0);
            setCol(3, origin, 1);
            ctx.transform = mulMat(m, ctx.transform);
            break;
        }
        case "matrix": {
            checkAttributes(node, ["value"]);
            const tokens = (node.getAttribute("value") ?? "").trim().split(/\s+/);
            if (tokens.length !== 16 && tokens.length !== 9) throw new RuntimeError("Mitsuba: matrix needs 9 or 16 values");
            const m = float4x4.identity();
            // Upstream writes mat[j][i] = tokens[i * n + j] against row-indexed
            // matrices, i.e. it reads the value list column-major. Mirrored as-is.
            const n = tokens.length === 16 ? 4 : 3;
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) m.set(j, i, parseFloatStrict(tokens[i * n + j]!, "floating point value"));
            }
            ctx.transform = mulMat(m, ctx.transform);
            break;
        }
        default:
            throw new RuntimeError(`Mitsuba: unknown tag '${node.nodeName}'`);
    }

    for (const child of Array.from(node.children)) parseXML(ctx, child, tag, props, counter, depth + 1);

    if (tag === "transform") props.set("transform", name, ctx.transform);
    return ["", ""];
}

/** Mirrors buildFrame's tangent when lookat has a degenerate up vector. */
function buildFrameTangent(n: float3): float3 {
    const sign = n.z >= 0 ? 1 : -1;
    const a = -1 / (sign + n.z);
    const b = n.x * n.y * a;
    return new float3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x);
}

/** Mirrors math::extractEulerAngleXYZ (radians), used for the env map rotation. */
function extractEulerAngleXYZ(m: float4x4): float3 {
    const at = (r: number, c: number) => m.get(r, c);
    const t1 = Math.atan2(at(1, 2), at(2, 2));
    const c2 = Math.hypot(at(0, 0), at(0, 1));
    const t2 = Math.atan2(-at(0, 2), c2);
    const s1 = Math.sin(t1);
    const c1 = Math.cos(t1);
    const t3 = Math.atan2(s1 * at(2, 0) - c1 * at(1, 0), c1 * at(1, 1) - s1 * at(2, 1));
    return new float3(-t1, -t2, -t3);
}

// -------------------------------------------------------------------------
// Building (mirrors MitsubaImporter.cpp)
// -------------------------------------------------------------------------

/** Swaps y and z; upstream applies it to disks and rectangles. */
const kTransformYtoZ = (() => {
    const m = float4x4.identity();
    m.set(1, 1, 0);
    m.set(1, 2, 1);
    m.set(2, 1, 1);
    m.set(2, 2, 0);
    return m;
})();
const kFlipZ = matrixFromScaling(new float3(1, 1, -1));

interface TextureInfo {
    /** Constant value, used when no texture is bound. */
    value: float4;
    /** Bitmap file to load into the slot. */
    path?: string;
    /** Already-decoded image (the procedural checkerboard). */
    bitmap?: ImageBitmap;
    srgb?: boolean;
    transform: float4x4;
}

interface BSDFInfo {
    material: MaterialBridge | null;
    /** What the material's base color was set to, for the area-emitter handover. */
    baseColor: float4;
    isDiffuse: boolean;
}

class MitsubaScene {
    private builder = new SceneBuilderBridge();
    private instances = new Map<string, XMLObject>();
    private warned = new Set<string>();

    constructor(
        private device: Device,
        private baseUrl: string,
    ) {}

    private warnOnce(msg: string): void {
        if (this.warned.has(msg)) return;
        this.warned.add(msg);
        Logger.warning(`MitsubaImporter: ${msg}`);
    }
    private unsupportedParameter(name: string): void {
        this.warnOnce(`Parameter '${name}' is not supported.`);
    }
    private unsupportedType(name: string): void {
        this.warnOnce(`Type '${name}' is not supported.`);
    }

    async load(source: string): Promise<Scene> {
        const doc = new DOMParser().parseFromString(source, "text/xml");
        const error = doc.getElementsByTagName("parsererror")[0];
        if (error) throw new RuntimeError(`Mitsuba: failed to parse XML: ${error.textContent ?? ""}`);
        const root = doc.documentElement;
        if (!root) throw new RuntimeError("Mitsuba: empty document");

        const ctx: ParseContext = { instances: this.instances, idCounter: 0, transform: float4x4.identity() };
        const [, sceneID] = parseXML(ctx, root, null, new Properties(), { arg: 0 });
        const scene = this.instances.get(sceneID);
        if (!scene || scene.cls !== "scene") throw new RuntimeError("Mitsuba: the root element must be a <scene>");

        await this.buildScene(scene);
        return this.builder.resolve(this.device, this.baseUrl);
    }

    /** Mirrors lookupIOR: a float property or a named material from the IOR table. */
    private lookupIOR(props: Properties, name: string, defaultIOR: string): number {
        if (props.has("float", name)) return props.getFloat(name);
        const material = props.getString(name, defaultIOR).toLowerCase();
        const ior = kIORTable[material];
        if (ior === undefined) {
            Logger.warning(`MitsubaImporter: '${material}' is not a valid IOR name.`);
            return 0;
        }
        return ior;
    }

    private async buildTexture(inst: XMLObject): Promise<TextureInfo> {
        const props = inst.props;
        // to_uv maps uv -> texture space, so the texture transform is its inverse;
        // the port has no per-material texture transform, so it only warns below.
        const toUV = props.getTransform("to_uv", float4x4.identity());

        if (inst.type === "bitmap") {
            const filename = props.getString("filename");
            const raw = props.getBool("raw", false);
            if (props.has("string", "filter_type")) this.unsupportedParameter("filter_type");
            if (props.has("string", "wrap_mode")) this.unsupportedParameter("wrap_mode");
            if (raw) this.warnOnce("Raw (linear) bitmaps load as sRGB; the port decides color space per texture slot.");
            return { value: new float4(1, 1, 1, 1), path: filename, srgb: !raw, transform: toUV };
        }
        if (inst.type === "checkerboard") {
            const color0 = props.getColor3("color0", new float3(0.4, 0.4, 0.4));
            const color1 = props.getColor3("color1", new float3(0.2, 0.2, 0.2));
            const kSize = 512;
            const pixels = new Uint8ClampedArray(kSize * kSize * 4);
            const quantize = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
            for (let y = 0; y < kSize; y++) {
                for (let x = 0; x < kSize; x++) {
                    const c = (x < kSize / 2) !== (y < kSize / 2) ? color1 : color0;
                    pixels.set([quantize(c.x), quantize(c.y), quantize(c.z), 255], (y * kSize + x) * 4);
                }
            }
            const bitmap = await createImageBitmap(new ImageData(pixels, kSize, kSize), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
            return { value: new float4(1, 1, 1, 1), bitmap, srgb: false, transform: toUV };
        }
        // Unsupported: mesh_attribute, volume
        this.unsupportedType(inst.type);
        return { value: new float4(1, 1, 1, 1), transform: float4x4.identity() };
    }

    /** Mirrors lookupTexture: a float, a color, a referenced texture, or the default. */
    private async lookupTexture(props: Properties, name: string, defaultValue: float4): Promise<TextureInfo> {
        if (props.has("float", name)) {
            const v = props.getFloat(name);
            return { value: new float4(v, v, v, v), transform: float4x4.identity() };
        }
        if (props.has("color3", name)) {
            const c = props.getColor3(name);
            return { value: new float4(c.x, c.y, c.z, 1), transform: float4x4.identity() };
        }
        if (props.hasRef(name)) {
            const inst = this.instances.get(props.getRef(name));
            if (!inst || inst.cls !== "texture") throw new RuntimeError(`Mitsuba: parameter '${name}' needs to be a color or texture`);
            return this.buildTexture(inst);
        }
        return { value: defaultValue, transform: float4x4.identity() };
    }

    /** Binds a texture slot, or falls back to the constant value. */
    private bindTexture(material: MaterialBridge, slot: string, texture: TextureInfo): boolean {
        if (texture.path) {
            material.loadTexture(slot, texture.path);
        } else if (texture.bitmap) {
            material.loadTextureBitmap(slot, texture.bitmap, texture.srgb ?? false);
        } else {
            return false;
        }
        this.unsupportedParameter("to_uv");
        return true;
    }

    /** Mirrors setMicrofacetProperties: roughness = sqrt(alpha). */
    private async setMicrofacetProperties(material: MaterialBridge, props: Properties, defaultAlpha = 0.1): Promise<void> {
        if (props.has("string", "distribution")) this.unsupportedParameter("distribution");
        if (props.has("boolean", "sample_visible")) this.unsupportedParameter("sample_visible");
        const alpha = await this.lookupTexture(props, "alpha", new float4(defaultAlpha, defaultAlpha, defaultAlpha, defaultAlpha));
        const textured = alpha.path !== undefined || alpha.bitmap !== undefined;
        if (textured) this.warnOnce("Microfacet alpha texture is not supported.");
        material.roughness = Math.sqrt(textured ? defaultAlpha : alpha.value.x);
    }

    private async buildBSDF(inst: XMLObject): Promise<BSDFInfo> {
        const props = inst.props;
        const none: BSDFInfo = { material: null, baseColor: new float4(1, 1, 1, 1), isDiffuse: false };

        switch (inst.type) {
            case "diffuse": {
                const material = new MaterialBridge(MaterialType.PBRTDiffuse, inst.id);
                const reflectance = await this.lookupTexture(props, "reflectance", new float4(0.5, 0.5, 0.5, 0.5));
                if (!this.bindTexture(material, "BaseColor", reflectance)) material.baseColor = reflectance.value;
                return { material, baseColor: reflectance.value, isDiffuse: true };
            }
            case "dielectric":
            case "roughdielectric": {
                const material = new MaterialBridge(MaterialType.Standard, inst.id);
                const intIOR = this.lookupIOR(props, "int_ior", "bk7");
                const extIOR = this.lookupIOR(props, "ext_ior", "air");
                if (props.has("float", "specular_reflectance")) this.unsupportedParameter("specular_reflectance");
                if (props.has("float", "specular_transmittance")) this.unsupportedParameter("specular_transmittance");
                material.specularTransmission = 1;
                material.doubleSided = true;
                material.roughness = 0;
                material.indexOfRefraction = intIOR / extIOR;
                if (inst.type === "roughdielectric") await this.setMicrofacetProperties(material, props);
                return { material, baseColor: new float4(1, 1, 1, 1), isDiffuse: false };
            }
            case "thindielectric": {
                const material = new MaterialBridge(MaterialType.Standard, inst.id);
                const intIOR = this.lookupIOR(props, "int_ior", "bk7");
                const extIOR = this.lookupIOR(props, "ext_ior", "air");
                if (props.has("float", "specular_reflectance")) this.unsupportedParameter("specular_reflectance");
                if (props.has("float", "specular_transmittance")) this.unsupportedParameter("specular_transmittance");
                material.specularTransmission = 1;
                material.doubleSided = true;
                material.thinSurface = true;
                material.roughness = 0;
                material.indexOfRefraction = intIOR / extIOR;
                return { material, baseColor: new float4(1, 1, 1, 1), isDiffuse: false };
            }
            case "conductor":
            case "roughconductor": {
                const material = new MaterialBridge(MaterialType.PBRTConductor, inst.id);
                if (props.has("float", "specular_reflectance")) this.unsupportedParameter("specular_reflectance");
                let baseColor = new float4(1, 1, 1, 1);
                if (props.has("color3", "eta") && props.has("color3", "k")) {
                    const eta = props.getColor3("eta");
                    baseColor = new float4(eta.x, eta.y, eta.z, 1);
                    material.baseColor = baseColor;
                    material.transmissionColor = props.getColor3("k");
                }
                material.roughness = { x: 0, y: 0 };
                material.doubleSided = true;
                if (inst.type === "roughconductor") {
                    const defaultAlpha = 0.1;
                    const alpha = await this.lookupTexture(props, "alpha", new float4(defaultAlpha, defaultAlpha, defaultAlpha, defaultAlpha));
                    const textured = alpha.path !== undefined || alpha.bitmap !== undefined;
                    if (textured) this.warnOnce("Microfacet alpha texture is not supported.");
                    material.roughness = textured ? { x: defaultAlpha, y: defaultAlpha } : { x: alpha.value.x, y: alpha.value.y };
                }
                return { material, baseColor, isDiffuse: false };
            }
            case "plastic":
            case "roughplastic": {
                const material = new MaterialBridge(MaterialType.Standard, inst.id);
                const diffuse = await this.lookupTexture(props, "diffuse_reflectance", new float4(0.5, 0.5, 0.5, 0.5));
                if (!this.bindTexture(material, "BaseColor", diffuse)) material.baseColor = diffuse.value;
                const intIOR = this.lookupIOR(props, "int_ior", "polypropylene");
                const extIOR = this.lookupIOR(props, "ext_ior", "air");
                if (props.has("boolean", "nonlinear")) this.unsupportedParameter("nonlinear");
                if (props.has("float", "specular_reflectance")) this.unsupportedParameter("specular_reflectance");
                material.roughness = 0;
                material.indexOfRefraction = intIOR / extIOR;
                if (inst.type === "roughplastic") await this.setMicrofacetProperties(material, props);
                return { material, baseColor: diffuse.value, isDiffuse: false };
            }
            case "twosided": {
                let inner: BSDFInfo | null = null;
                for (const [, id] of props.namedReferences()) {
                    const child = this.instances.get(id);
                    if (!child || child.cls !== "bsdf") continue;
                    if (inner) throw new RuntimeError("Mitsuba: 'twosided' BSDF can only have one nested BSDF");
                    inner = await this.buildBSDF(child);
                }
                // Upstream marks the (still null) outer material double-sided here,
                // which crashes; the inner material is what it means (docs §9).
                if (inner?.material) inner.material.doubleSided = true;
                return inner ?? none;
            }
            default:
                this.unsupportedType(inst.type);
                return none;
        }
    }

    /** Mirrors buildMedium: homogeneous media become volume properties. */
    private buildMedium(inst: XMLObject): { sigmaS: float3; sigmaA: float3 } | null {
        const props = inst.props;
        if (inst.type !== "homogeneous") {
            this.unsupportedType(inst.type); // heterogeneous
            return null;
        }
        const scale = props.getFloat("scale", 1);
        const mul = (v: float3, s: number) => new float3(v.x * s, v.y * s, v.z * s);
        if (props.has("string", "material")) {
            this.unsupportedParameter("material");
            return null;
        }
        if (props.has("color3", "sigma_s") && props.has("color3", "sigma_a")) {
            return { sigmaS: mul(props.getColor3("sigma_s"), scale), sigmaA: mul(props.getColor3("sigma_a"), scale) };
        }
        if (props.has("color3", "albedo") && props.has("color3", "sigma_t")) {
            const albedo = props.getColor3("albedo");
            // Upstream reads sigma_s here rather than sigma_t; mirrored.
            const sigmaT = props.getColor3("sigma_s", new float3(0, 0, 0));
            const sigmaS = new float3(albedo.x * sigmaT.x * scale, albedo.y * sigmaT.y * scale, albedo.z * sigmaT.z * scale);
            return { sigmaS, sigmaA: new float3(sigmaT.x * scale - sigmaS.x, sigmaT.y * scale - sigmaS.y, sigmaT.z * scale - sigmaS.z) };
        }
        return null;
    }

    private async buildShape(inst: XMLObject): Promise<{ mesh: TriangleMeshDesc | null; transform: float4x4; material: MaterialBridge | null }> {
        const props = inst.props;
        const toWorld = props.getTransform("to_world", float4x4.identity());
        if (props.has("boolean", "flip_normals")) this.unsupportedParameter("flip_normals");

        let mesh: TriangleMeshDesc | null = null;
        let transform = toWorld;

        switch (inst.type) {
            case "obj":
            case "ply": {
                const filename = props.getString("filename");
                const faceNormals = props.getBool("face_normals", false);
                if (props.has("boolean", "flip_tex_coords")) this.unsupportedParameter("flip_tex_coords");
                // Upstream picks GenSmoothNormals unless the shape asks for face normals.
                mesh = TriangleMesh.createFromFile(filename, !faceNormals);
                break;
            }
            case "sphere": {
                const center = props.getFloat3("center", new float3(0, 0, 0));
                mesh = TriangleMesh.createSphere(props.getFloat("radius", 1));
                transform = mulMat(toWorld, matrixFromTranslation(center));
                break;
            }
            case "disk":
                mesh = TriangleMesh.createDisk(1);
                transform = mulMat(toWorld, kTransformYtoZ);
                break;
            case "rectangle":
                mesh = TriangleMesh.createQuad(new float2(2, 2));
                transform = mulMat(toWorld, kTransformYtoZ);
                break;
            case "cube":
                mesh = TriangleMesh.createCube(new float3(2, 2, 2));
                break;
            default:
                this.unsupportedType(inst.type);
        }

        // Nested BSDF.
        let bsdf: BSDFInfo | null = null;
        for (const [, id] of props.namedReferences()) {
            const child = this.instances.get(id);
            if (!child || child.cls !== "bsdf") continue;
            if (bsdf?.material) throw new RuntimeError("Mitsuba: shape can only have one BSDF");
            bsdf = await this.buildBSDF(child);
        }
        let material = bsdf?.material ?? null;
        let isDiffuse = bsdf?.isDiffuse ?? false;
        let baseColor = bsdf?.baseColor ?? new float4(1, 1, 1, 1);
        if (!material) {
            material = new MaterialBridge(MaterialType.Standard, inst.id);
            isDiffuse = false;
            baseColor = new float4(1, 1, 1, 1);
        }

        // Interior medium.
        for (const [name, id] of props.namedReferences()) {
            const child = this.instances.get(id);
            if (!child || child.cls !== "medium" || name !== "interior") continue;
            const medium = this.buildMedium(child);
            if (medium) {
                material.volumeScattering = medium.sigmaS;
                material.volumeAbsorption = medium.sigmaA;
            }
        }

        // Nested area emitter: the shape becomes an emissive StandardMaterial.
        for (const [, id] of props.namedReferences()) {
            const child = this.instances.get(id);
            if (!child || child.cls !== "emitter" || child.type !== "area") continue;
            if (!isDiffuse) throw new RuntimeError("Mitsuba: shape with area emitter must have a diffuse material");
            const radiance = child.props.getColor3("radiance");
            const factor = Math.max(radiance.x, radiance.y, radiance.z);
            const emissive = factor > 0 ? new float3(radiance.x / factor, radiance.y / factor, radiance.z / factor) : radiance;
            material = new MaterialBridge(MaterialType.Standard, inst.id);
            material.emissiveColor = emissive;
            material.emissiveFactor = factor;
            material.metallic = 0;
            material.roughness = 1;
            material.baseColor = baseColor;
        }

        return { mesh, transform, material };
    }

    /** Mirrors buildSensor: a perspective/thinlens sensor becomes the camera. */
    private buildSensor(inst: XMLObject): { camera: CameraBridge; transform: float4x4 } | null {
        const props = inst.props;
        const toWorld = props.getTransform("to_world", float4x4.identity());

        // The film's resolution feeds the fov -> focal length mapping.
        let width = 768;
        let height = 576;
        for (const [, id] of props.namedReferences()) {
            const child = this.instances.get(id);
            if (!child || child.cls !== "film") continue;
            width = child.props.getInt("width", 768);
            height = child.props.getInt("height", 576);
        }

        if (inst.type !== "perspective" && inst.type !== "thinlens") {
            // Unsupported: orthographic, radiancemeter, irradiancemeter, distant, batch
            this.unsupportedType(inst.type);
            return null;
        }
        if (props.has("float", "focal_length") && props.has("float", "fov")) {
            throw new RuntimeError("Mitsuba: cannot specify both 'focal_length' and 'fov'");
        }
        let focalLength = props.getFloat("focal_length", 50);
        if (props.has("float", "fov")) {
            // Upstream passes the film *width* as the frame height here.
            const filmWidth = (24 / height) * width;
            focalLength = fovYToFocalLength(radians(props.getFloat("fov")), filmWidth);
        }
        if (props.has("string", "fov_axis")) this.unsupportedParameter("fov_axis");

        const camera = new CameraBridge();
        camera.focalLength = focalLength;
        camera.nearPlane = props.getFloat("near_clip", 1e-2);
        camera.farPlane = props.getFloat("far_clip", 1e4);
        camera.focalDistance = props.getFloat("focus_distance", 1);
        camera.apertureRadius = props.getFloat("aperture_radius", 0);
        return { camera, transform: mulMat(toWorld, kFlipZ) };
    }

    /** Mirrors buildEmitter for the scene-level emitters (area is handled by the shape). */
    private buildEmitter(inst: XMLObject): void {
        const props = inst.props;
        let toWorld = props.getTransform("to_world", float4x4.identity());

        if (inst.type === "area") throw new RuntimeError("Mitsuba: 'area' emitter needs to be nested in a shape");
        if (inst.type === "constant") {
            const radiance = props.getColor3("radiance");
            if (this.builder.envMap) throw new RuntimeError("Mitsuba: only one envmap can be added");
            this.builder.envMap = { path: "", intensity: 1, constantColor: [radiance.x, radiance.y, radiance.z] };
            return;
        }
        if (inst.type === "envmap") {
            if (this.builder.envMap) throw new RuntimeError("Mitsuba: only one envmap can be added");
            toWorld = mulMat(toWorld, kFlipZ);
            const rotation = extractEulerAngleXYZ(toWorld);
            this.builder.envMap = {
                path: props.getString("filename"),
                intensity: props.getFloat("scale", 1),
                rotation: { x: degrees(rotation.x), y: degrees(rotation.y), z: degrees(rotation.z) },
            };
            return;
        }
        if (inst.type === "point" && props.has("float3", "position") && props.has("transform", "to_world")) {
            throw new RuntimeError("Mitsuba: either 'to_world' or 'position' can be specified, not both");
        }
        // Unsupported upstream: point, spot, projector, directional, directionalarea
        this.unsupportedType(inst.type);
    }

    private async buildScene(scene: XMLObject): Promise<void> {
        for (const [, id] of scene.props.namedReferences()) {
            const child = this.instances.get(id);
            if (!child) continue;
            switch (child.cls) {
                case "sensor": {
                    const sensor = this.buildSensor(child);
                    if (!sensor) break;
                    // The port's camera bridge takes a pose, so the node transform
                    // is applied here: Falcor looks down -z in camera space.
                    sensor.camera.position = transformPoint(sensor.transform, new float3(0, 0, 0));
                    sensor.camera.target = transformPoint(sensor.transform, new float3(0, 0, -1));
                    sensor.camera.up = normalize3(transformVector(sensor.transform, new float3(0, 1, 0)));
                    this.builder.addCamera(sensor.camera);
                    break;
                }
                case "emitter":
                    this.buildEmitter(child);
                    break;
                case "shape": {
                    const shape = await this.buildShape(child);
                    if (!shape.mesh || !shape.material) break;
                    const meshID = this.builder.addTriangleMesh(shape.mesh, shape.material);
                    const nodeID = this.builder.addNode(child.id, shape.transform);
                    this.builder.addMeshInstance(nodeID, meshID);
                    break;
                }
                default:
                    break; // integrator/sampler/film/rfilter carry no scene data
            }
        }
    }
}

/**
 * Parses a Mitsuba 3 XML scene and builds a Scene (assets fetched relative to
 * baseUrl). Parallels runPbrtScene for `.pbrt` and runSceneScript for `.pyscene`.
 */
export async function runMitsubaScene(device: Device, source: string, baseUrl: string): Promise<Scene> {
    return withScriptSearchPath(baseUrl, () => new MitsubaScene(device, baseUrl).load(source));
}
