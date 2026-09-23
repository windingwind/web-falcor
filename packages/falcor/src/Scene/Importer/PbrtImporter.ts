/**
 * pbrt-v4 scene importer (Scene/Importer) — mirrors upstream Falcor's
 * plugins/importers/PBRTImporter, which itself supports only a limited subset
 * of pbrt-v4. A `.pbrt` file is a *whole scene* (camera + lights + geometry),
 * so unlike the mesh-only glTF/FBX importers this drives a full SceneBuilder
 * and produces a Scene, more like a .pyscene.
 *
 * Coverage (matching the upstream importer's README):
 *   - Camera: perspective (fov, lensradius, focaldistance)
 *   - Materials -> StandardMaterial: diffuse, coateddiffuse, conductor,
 *     coatedconductor, dielectric, thindielectric, diffusetransmission
 *   - Shapes: trianglemesh, plymesh, sphere, disk
 *   - Lights: distant (-> DirectionalLight), infinite w/ filename (-> env map),
 *     area light diffuse (-> emissive StandardMaterial)
 *   - Transforms: Transform/ConcatTransform/Translate/Rotate/Scale/LookAt/Identity
 *   - Structure: WorldBegin, AttributeBegin/End, TransformBegin/End,
 *     MakeNamedMaterial/NamedMaterial/Material, AreaLightSource,
 *     ReverseOrientation, Include/Import
 *
 * Coordinate handling follows upstream exactly: geometry is kept in pbrt's
 * (left-handed) coordinates verbatim; only the camera is z-flipped (kInvertZ)
 * to look correctly in Falcor's right-handed space.
 *
 * Materials map to StandardMaterial by default; with the Settings option
 * `PBRTImporter:usePBRTMaterials` (same key as native) diffuse/coateddiffuse/
 * conductor/coatedconductor/dielectric/diffusetransmission map to the
 * dedicated PBRT material classes instead (area lights stay Standard, like
 * native). Spectral parameters (named, sampled and blackbody) convert to RGB
 * through the CIE curves exactly as native does (Utils/Color/Spectrum.ts), and
 * `infinite` lights take native's two paths: a constant `L` becomes a one-pixel
 * env map, a `filename` an equal-area-octahedral map converted to lat-long.
 * Documented divergence: non-constant/anisotropic roughness is not supported.
 */

import type { Device } from "../../Core/API/Device.js";
import type { Scene } from "../Scene.js";
import type { StaticVertex } from "../SceneData.js";
import { LightType } from "../SceneData.js";
import { MaterialType } from "../Material/MaterialData.js";
import { float2, float3, float4, normalize3, sub3, cross, dot3 } from "../../Utils/Math/Vector.js";
import {
    float4x4,
    mulMat,
    inverse,
    matrixFromTranslation,
    matrixFromScaling,
    matrixFromRotationAxisAngle,
    transformPoint,
    transformVector,
    extractEulerAngleXYZ,
} from "../../Utils/Math/Matrix.js";
import { fovYToFocalLength } from "../Camera/Camera.js";
import { BlackbodySpectrum, PiecewiseLinearSpectrum, Spectra, spectrumToRGB } from "../../Utils/Color/Spectrum.js";
import { RuntimeError } from "../../Core/Error.js";
import { AssetCategory, resolveAssetUrl, withScriptSearchPath } from "../../Core/AssetResolver.js";
import { getGlobalSettings } from "../../Utils/Scripting/Scripting.js";
import {
    SceneBuilderBridge,
    CameraBridge,
    LightBridge,
    MaterialBridge,
    TriangleMesh,
    type TriangleMeshDesc,
} from "../SceneBuilder.js";

// pbrt is left-handed; Falcor is right-handed. Upstream flips only the camera's
// z axis (geometry is loaded verbatim). Mirrors PBRTImporter.cpp `kInvertZ`.
const kInvertZ = matrixFromScaling(new float3(1, 1, -1));
const radians = (deg: number) => (deg * Math.PI) / 180;

// -------------------------------------------------------------------------
// Tokenizer
// -------------------------------------------------------------------------

interface Token {
    kind: "string" | "lbracket" | "rbracket" | "word";
    text: string;
}

/** Splits pbrt source into tokens (quoted strings, brackets, bare words). */
function tokenize(src: string): Token[] {
    const tokens: Token[] = [];
    let i = 0;
    const n = src.length;
    while (i < n) {
        const c = src[i]!;
        if (c === "#") {
            // Comment to end of line.
            while (i < n && src[i] !== "\n") i++;
        } else if (c === " " || c === "\t" || c === "\r" || c === "\n") {
            i++;
        } else if (c === '"') {
            let j = i + 1;
            while (j < n && src[j] !== '"') j++;
            tokens.push({ kind: "string", text: src.slice(i + 1, j) });
            i = j + 1;
        } else if (c === "[") {
            tokens.push({ kind: "lbracket", text: "[" });
            i++;
        } else if (c === "]") {
            tokens.push({ kind: "rbracket", text: "]" });
            i++;
        } else {
            let j = i;
            while (j < n && !" \t\r\n[]\"#".includes(src[j]!)) j++;
            tokens.push({ kind: "word", text: src.slice(i, j) });
            i = j;
        }
    }
    return tokens;
}

// -------------------------------------------------------------------------
// Parameter lists  ("type name" [ values ])
// -------------------------------------------------------------------------

interface Param {
    type: string;
    values: string[];
}
type Params = Map<string, Param>;

/** Typed accessors over a parsed parameter dictionary. */
const P = {
    float(p: Params, name: string, def: number): number {
        const v = p.get(name);
        return v && v.values.length ? parseFloat(v.values[0]!) : def;
    },
    has(p: Params, name: string): boolean {
        return p.has(name);
    },
    bool(p: Params, name: string, def: boolean): boolean {
        const v = p.get(name);
        if (!v || !v.values.length) return def;
        return v.values[0] === "true";
    },
    string(p: Params, name: string, def = ""): string {
        const v = p.get(name);
        return v && v.values.length ? v.values[0]! : def;
    },
    ints(p: Params, name: string): number[] {
        const v = p.get(name);
        return v ? v.values.map((s) => Math.trunc(parseFloat(s))) : [];
    },
    floats(p: Params, name: string): number[] {
        const v = p.get(name);
        return v ? v.values.map(parseFloat) : [];
    },
    /**
     * rgb/color -> float3; single float -> gray; blackbody, named and sampled
     * spectra -> RGB through the CIE curves (mirrors getSpectrumAsRGB).
     */
    rgb(p: Params, name: string, def: float3): float3 {
        const v = p.get(name);
        if (!v || !v.values.length) return def;
        if (v.type === "rgb" || v.type === "color" || v.type === "vector3" || v.type === "point3") {
            const f = v.values.map(parseFloat);
            return new float3(f[0]!, f[1] ?? f[0]!, f[2] ?? f[0]!);
        }
        if (v.type === "float") {
            const g = parseFloat(v.values[0]!);
            return new float3(g, g, g);
        }
        return spectrumParamToRGB(v, name) ?? def;
    },
    /** Mirrors ParameterDictionary::hasSpectrum. */
    hasSpectrum(p: Params, name: string): boolean {
        const t = p.get(name)?.type;
        return t === "rgb" || t === "blackbody" || t === "spectrum";
    },
};

/**
 * Mirrors ParameterDictionary::extractSpectrumArray + spectrumToRGB for the
 * spectral parameter types: `blackbody [T]`, `spectrum "named"` and
 * `spectrum [lambda value ...]`. Returns undefined for anything else.
 */
function spectrumParamToRGB(v: Param, name: string): float3 | undefined {
    if (v.type === "blackbody") return spectrumToRGB(new BlackbodySpectrum(parseFloat(v.values[0]!)));
    if (v.type !== "spectrum") return undefined;
    const first = v.values[0]!;
    if (Number.isFinite(Number(first))) {
        const f = v.values.map(parseFloat);
        if (f.length % 2 !== 0) throw new RuntimeError(`pbrt: found odd number of values for '${name}'`);
        const wavelengths: number[] = [];
        const values: number[] = [];
        for (let i = 0; i < f.length; i += 2) {
            if (i > 0 && f[i]! <= wavelengths[wavelengths.length - 1]!) {
                throw new RuntimeError(`pbrt: spectrum '${name}' wavelengths aren't increasing`);
            }
            wavelengths.push(f[i]!);
            values.push(f[i + 1]!);
        }
        return spectrumToRGB(new PiecewiseLinearSpectrum(wavelengths, values));
    }
    const named = Spectra.getNamedSpectrum(first);
    // Native falls back to reading a spectrum file, which it leaves unimplemented.
    if (!named) throw new RuntimeError(`pbrt: unable to read spectrum file '${first}'`);
    return spectrumToRGB(named);
}

/** RGB of one of native's named spectra (they are always present). */
function namedSpectrumRGB(name: string): float3 {
    return spectrumToRGB(Spectra.getNamedSpectrum(name)!);
}

// -------------------------------------------------------------------------
// Material / light intermediate + conductor helpers
// -------------------------------------------------------------------------

interface MaterialDef {
    type: string;
    name: string;
    params: Params;
}

/** A `Texture` directive: `Texture "name" "spectrum|float" "class" params`. */
interface TextureDef {
    /** "spectrum" or "float". */
    kind: string;
    /** pbrt texture class: imagemap, constant, scale, mix, checkerboard, ... */
    textureClass: string;
    params: Params;
}

// Native's default conductor is copper: the "metal-Cu" eta/k spectra in RGB.
const copperEta = () => namedSpectrumRGB("metal-Cu-eta");
const copperK = () => namedSpectrumRGB("metal-Cu-k");

/** Fresnel reflectance of a conductor (mirrors fresnelDieletricConductor). */
function fresnelConductor(eta: float3, k: float3, cosTheta: number): float3 {
    const c2 = cosTheta * cosTheta;
    const s2 = 1 - c2;
    const out = (e: number, kk: number): number => {
        const eta2 = e * e;
        const k2 = kk * kk;
        const t0 = eta2 - k2 - s2;
        const a2b2 = Math.sqrt(Math.max(0, t0 * t0 + 4 * eta2 * k2));
        const t1 = a2b2 + c2;
        const a = Math.sqrt(Math.max(0, 0.5 * (a2b2 + t0)));
        const t2 = 2 * a * cosTheta;
        const Rs = (t1 - t2) / (t1 + t2);
        const t3 = c2 * a2b2 + s2 * s2;
        const t4 = t2 * s2;
        const Rp = (Rs * (t3 - t4)) / (t3 + t4);
        return 0.5 * (Rp + Rs);
    };
    return new float3(out(eta.x, k.x), out(eta.y, k.y), out(eta.z, k.z));
}

/** getRoughness -> NDF alpha (replicates upstream, incl. the u+v sum). */
function scalarRoughnessAlpha(p: Params, rName = "roughness", uName = "uroughness", vName = "vroughness"): number {
    const uc = P.has(p, uName) ? P.float(p, uName, 0) : P.float(p, rName, 0);
    const vc = P.has(p, vName) ? P.float(p, vName, 0) : P.float(p, rName, 0);
    // Native quirk kept: getRoughness builds float2{u + v} (a broadcast of the sum).
    let alpha = uc + vc;
    if (P.bool(p, "remaproughness", true)) alpha = Math.sqrt(alpha);
    return alpha;
}

/** Mirrors getScalarEta: a spectral eta collapses to the mean of its RGB. */
function scalarEta(p: Params, name = "eta"): number {
    if (P.hasSpectrum(p, name)) {
        const rgb = P.rgb(p, name, new float3(1.5, 1.5, 1.5));
        return (rgb.x + rgb.y + rgb.z) / 3;
    }
    return P.float(p, name, 1.5);
}

/** Mirrors getConductorEtaK: reflectance -> (eta=1, k from Fresnel inversion), else eta/k (copper default). */
function conductorEtaK(p: Params, rName = "reflectance", etaName = "eta", kName = "k"): { eta: float3; k: float3 } {
    if (P.has(p, rName)) {
        const r = P.rgb(p, rName, new float3(0.5, 0.5, 0.5));
        const cl = (x: number) => Math.min(0.9999, Math.max(0, x));
        const rc = new float3(cl(r.x), cl(r.y), cl(r.z));
        const k = new float3(
            (2 * Math.sqrt(rc.x)) / Math.sqrt(1 - rc.x),
            (2 * Math.sqrt(rc.y)) / Math.sqrt(1 - rc.y),
            (2 * Math.sqrt(rc.z)) / Math.sqrt(1 - rc.z),
        );
        return { eta: new float3(1, 1, 1), k };
    }
    return {
        eta: P.has(p, etaName) ? P.rgb(p, etaName, copperEta()) : copperEta(),
        k: P.has(p, kName) ? P.rgb(p, kName, copperK()) : copperK(),
    };
}

function conductorSpecularAlbedo(p: Params, rName = "reflectance", etaName = "eta", kName = "k"): float3 {
    const { eta, k } = conductorEtaK(p, rName, etaName, kName);
    return fresnelConductor(eta, k, 1);
}

// -------------------------------------------------------------------------
// Minimal PLY loader (ascii + binary_little_endian), for `plymesh`.
// -------------------------------------------------------------------------

const kPlyTypeSize: Record<string, number> = {
    char: 1, uchar: 1, int8: 1, uint8: 1,
    short: 2, ushort: 2, int16: 2, uint16: 2,
    int: 4, uint: 4, int32: 4, uint32: 4, float: 4, float32: 4,
    double: 8, float64: 8,
};

function parsePly(buf: ArrayBuffer): TriangleMeshDesc {
    const bytes = new Uint8Array(buf);
    // Read the ascii header (terminated by "end_header\n").
    let headerEnd = -1;
    for (let i = 0; i < bytes.length - 10; i++) {
        if (bytes[i] === 0x65 /*e*/ && String.fromCharCode(...bytes.slice(i, i + 10)) === "end_header") {
            let j = i + 10;
            while (j < bytes.length && bytes[j] !== 0x0a) j++;
            headerEnd = j + 1;
            break;
        }
    }
    if (headerEnd < 0) throw new RuntimeError("PLY: no end_header");
    const header = new TextDecoder().decode(bytes.slice(0, headerEnd));
    const lines = header.split("\n").map((l) => l.trim()).filter(Boolean);

    let format = "ascii";
    interface Prop { name: string; type: string; isList?: boolean; countType?: string; }
    interface Elem { name: string; count: number; props: Prop[]; }
    const elems: Elem[] = [];
    for (const line of lines) {
        const t = line.split(/\s+/);
        if (t[0] === "format") format = t[1]!;
        else if (t[0] === "element") elems.push({ name: t[1]!, count: parseInt(t[2]!), props: [] });
        else if (t[0] === "property") {
            const el = elems[elems.length - 1]!;
            if (t[1] === "list") el.props.push({ name: t[4]!, type: t[3]!, isList: true, countType: t[2]! });
            else el.props.push({ name: t[2]!, type: t[1]! });
        }
    }

    const positions: float3[] = [];
    const normals: float3[] = [];
    const uvs: float2[] = [];
    const indices: number[] = [];

    if (format === "ascii") {
        const body = new TextDecoder().decode(bytes.slice(headerEnd));
        const toks = body.split(/\s+/).filter(Boolean);
        let k = 0;
        for (const el of elems) {
            for (let e = 0; e < el.count; e++) {
                const vals: Record<string, number> = {};
                const face: number[] = [];
                for (const pr of el.props) {
                    if (pr.isList) {
                        const cnt = parseFloat(toks[k++]!);
                        for (let c = 0; c < cnt; c++) face.push(Math.trunc(parseFloat(toks[k++]!)));
                    } else vals[pr.name] = parseFloat(toks[k++]!);
                }
                consumeElem(el.name, vals, face);
            }
        }
    } else {
        const le = format !== "binary_big_endian";
        const dv = new DataView(buf, headerEnd);
        let off = 0;
        const read = (type: string): number => {
            let v = 0;
            switch (type) {
                case "char": case "int8": v = dv.getInt8(off); off += 1; break;
                case "uchar": case "uint8": v = dv.getUint8(off); off += 1; break;
                case "short": case "int16": v = dv.getInt16(off, le); off += 2; break;
                case "ushort": case "uint16": v = dv.getUint16(off, le); off += 2; break;
                case "int": case "int32": v = dv.getInt32(off, le); off += 4; break;
                case "uint": case "uint32": v = dv.getUint32(off, le); off += 4; break;
                case "float": case "float32": v = dv.getFloat32(off, le); off += 4; break;
                case "double": case "float64": v = dv.getFloat64(off, le); off += 8; break;
                default: off += kPlyTypeSize[type] ?? 4; break;
            }
            return v;
        };
        for (const el of elems) {
            for (let e = 0; e < el.count; e++) {
                const vals: Record<string, number> = {};
                const face: number[] = [];
                for (const pr of el.props) {
                    if (pr.isList) {
                        const cnt = read(pr.countType!);
                        for (let c = 0; c < cnt; c++) face.push(read(pr.type));
                    } else vals[pr.name] = read(pr.type);
                }
                consumeElem(el.name, vals, face);
            }
        }
    }

    function consumeElem(elem: string, vals: Record<string, number>, face: number[]): void {
        if (elem === "vertex") {
            positions.push(new float3(vals.x ?? 0, vals.y ?? 0, vals.z ?? 0));
            if ("nx" in vals) normals.push(new float3(vals.nx!, vals.ny ?? 0, vals.nz ?? 0));
            const u = vals.u ?? vals.s;
            const v = vals.v ?? vals.t;
            if (u !== undefined) uvs.push(new float2(u, v ?? 0));
        } else if (elem === "face") {
            // Fan-triangulate polygons.
            for (let f = 2; f < face.length; f++) indices.push(face[0]!, face[f - 1]!, face[f]!);
        }
    }

    return assembleMesh(positions, indices, normals.length === positions.length ? normals : [], uvs.length === positions.length ? uvs : []);
}

// -------------------------------------------------------------------------
// Mesh assembly (shared by trianglemesh + plymesh)
// -------------------------------------------------------------------------

const kZeroTangent = new float4(0, 0, 0, 0);

/** Builds a TriangleMeshDesc; computes smooth normals if none are supplied. */
function assembleMesh(positions: float3[], indices: number[], normals: float3[], uvs: float2[]): TriangleMeshDesc {
    let N = normals;
    if (N.length !== positions.length) {
        // Area-weighted smooth normals.
        const acc = positions.map(() => new float3(0, 0, 0));
        for (let i = 0; i + 2 < indices.length; i += 3) {
            const a = positions[indices[i]!]!;
            const b = positions[indices[i + 1]!]!;
            const c = positions[indices[i + 2]!]!;
            const n = cross(sub3(b, a), sub3(c, a));
            for (const idx of [indices[i]!, indices[i + 1]!, indices[i + 2]!]) {
                acc[idx]!.x += n.x; acc[idx]!.y += n.y; acc[idx]!.z += n.z;
            }
        }
        N = acc.map((n) => {
            const l = Math.hypot(n.x, n.y, n.z);
            return l > 0 ? new float3(n.x / l, n.y / l, n.z / l) : new float3(0, 1, 0);
        });
    }
    const vertices: StaticVertex[] = positions.map((p, i) => ({
        position: p,
        normal: N[i]!,
        tangent: kZeroTangent,
        texCrd: uvs[i] ?? new float2(0, 0),
    }));
    return { vertices, indices: new Uint32Array(indices) };
}

// -------------------------------------------------------------------------
// Parser / interpreter
// -------------------------------------------------------------------------

interface GfxState {
    ctm: float4x4;
    reverseOrientation: boolean;
    material: MaterialDef | null; // current (named or inline)
    areaLight: Params | null;
}

class PbrtScene {
    private builder = new SceneBuilderBridge();
    private namedMaterials = new Map<string, MaterialDef>();
    /** Named `Texture` declarations, referenced by material parameters. */
    private textures = new Map<string, TextureDef>();
    private matCache = new Map<MaterialDef, MaterialBridge>();
    private state: GfxState = { ctm: float4x4.identity(), reverseOrientation: false, material: null, areaLight: null };
    private stack: GfxState[] = [];
    private cameraFromWorld: float4x4 | null = null;
    private cameraParams: Params | null = null;
    private warned = new Set<string>();

    constructor(private device: Device, private baseUrl: string) {}

    private warn(msg: string): void {
        if (!this.warned.has(msg)) {
            this.warned.add(msg);
            console.warn(`[pbrt] ${msg}`);
        }
    }

    async load(source: string): Promise<Scene> {
        await this.run(tokenize(source));
        // Apply the captured camera now that all transforms are known.
        if (this.cameraFromWorld && this.cameraParams) this.applyCamera(this.cameraFromWorld, this.cameraParams);
        return this.builder.resolve(this.device, this.baseUrl);
    }

    /** Reads a parameter list starting at cursor `c` (a mutable {i}). */
    private readParams(tokens: Token[], c: { i: number }): Params {
        const params: Params = new Map();
        while (c.i < tokens.length && tokens[c.i]!.kind === "string" && tokens[c.i]!.text.includes(" ")) {
            const decl = tokens[c.i++]!.text.trim().split(/\s+/);
            const type = decl[0]!;
            const name = decl.slice(1).join(" ");
            const values: string[] = [];
            if (c.i < tokens.length && tokens[c.i]!.kind === "lbracket") {
                c.i++;
                while (c.i < tokens.length && tokens[c.i]!.kind !== "rbracket") values.push(tokens[c.i++]!.text);
                c.i++; // consume ']'
            } else if (c.i < tokens.length) {
                values.push(tokens[c.i++]!.text);
            }
            params.set(name, { type, values });
        }
        return params;
    }

    private nextString(tokens: Token[], c: { i: number }): string {
        const t = tokens[c.i];
        if (!t || t.kind !== "string") throw new RuntimeError(`pbrt: expected string at token ${c.i}`);
        c.i++;
        return t.text;
    }

    private nextFloats(tokens: Token[], c: { i: number }, count: number): number[] {
        const out: number[] = [];
        // Optionally wrapped in brackets.
        const bracketed = tokens[c.i]?.kind === "lbracket";
        if (bracketed) c.i++;
        while (out.length < count && c.i < tokens.length && tokens[c.i]!.kind === "word") out.push(parseFloat(tokens[c.i++]!.text));
        if (bracketed && tokens[c.i]?.kind === "rbracket") c.i++;
        return out;
    }

    private async run(tokens: Token[]): Promise<void> {
        const c = { i: 0 };
        while (c.i < tokens.length) {
            const tok = tokens[c.i++]!;
            if (tok.kind !== "word") continue; // stray value
            const d = tok.text;
            switch (d) {
                // --- Options block (mostly ignored) ---
                case "Integrator": case "Sampler": case "PixelFilter": case "Film":
                case "Accelerator": case "ColorSpace": case "Option": case "Filter":
                    if (tokens[c.i]?.kind === "string") this.nextString(tokens, c);
                    this.readParams(tokens, c);
                    break;
                case "Camera": {
                    this.cameraParams = new Map();
                    const type = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    if (type !== "perspective") this.warn(`unsupported camera type '${type}'`);
                    this.cameraFromWorld = this.state.ctm.clone();
                    this.cameraParams = params;
                    break;
                }
                // --- Transforms ---
                case "Identity": this.state.ctm = float4x4.identity(); break;
                case "Transform": this.state.ctm = this.readMatrix(tokens, c); break;
                case "ConcatTransform": this.state.ctm = mulMat(this.state.ctm, this.readMatrix(tokens, c)); break;
                case "Translate": {
                    const [x, y, z] = this.nextFloats(tokens, c, 3);
                    this.state.ctm = mulMat(this.state.ctm, matrixFromTranslation(new float3(x!, y!, z!)));
                    break;
                }
                case "Scale": {
                    const [x, y, z] = this.nextFloats(tokens, c, 3);
                    this.state.ctm = mulMat(this.state.ctm, matrixFromScaling(new float3(x!, y!, z!)));
                    break;
                }
                case "Rotate": {
                    const [a, x, y, z] = this.nextFloats(tokens, c, 4);
                    this.state.ctm = mulMat(this.state.ctm, matrixFromRotationAxisAngle(radians(a!), new float3(x!, y!, z!)));
                    break;
                }
                case "LookAt": {
                    const f = this.nextFloats(tokens, c, 9);
                    this.state.ctm = mulMat(this.state.ctm, lookAtLH(new float3(f[0]!, f[1]!, f[2]!), new float3(f[3]!, f[4]!, f[5]!), new float3(f[6]!, f[7]!, f[8]!)));
                    break;
                }
                case "CoordinateSystem": case "CoordSysTransform": this.nextString(tokens, c); break;
                // --- State scoping ---
                case "WorldBegin": this.state.ctm = float4x4.identity(); break;
                case "WorldEnd": break;
                case "AttributeBegin": case "TransformBegin": case "ObjectBegin":
                    if (d === "ObjectBegin") { this.nextString(tokens, c); this.warn("ObjectBegin/instancing not supported; shapes are inlined"); }
                    this.stack.push({ ...this.state, ctm: this.state.ctm.clone() });
                    break;
                case "AttributeEnd": case "TransformEnd": case "ObjectEnd": {
                    const s = this.stack.pop();
                    if (s) this.state = s;
                    break;
                }
                case "ObjectInstance": this.nextString(tokens, c); break;
                case "ReverseOrientation": this.state.reverseOrientation = !this.state.reverseOrientation; break;
                case "Attribute": this.nextString(tokens, c); this.readParams(tokens, c); break;
                // --- Materials ---
                case "MakeNamedMaterial": {
                    const name = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    this.namedMaterials.set(name, { type: P.string(params, "type", "diffuse"), name, params });
                    break;
                }
                case "Material": {
                    const type = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    this.state.material = { type, name: "", params };
                    break;
                }
                case "NamedMaterial": {
                    const name = this.nextString(tokens, c);
                    const m = this.namedMaterials.get(name);
                    if (!m) this.warn(`unknown named material '${name}'`);
                    this.state.material = m ?? null;
                    break;
                }
                case "Texture": {
                    const name = this.nextString(tokens, c);
                    const kind = this.nextString(tokens, c);
                    const textureClass = this.nextString(tokens, c);
                    this.textures.set(name, { kind, textureClass, params: this.readParams(tokens, c) });
                    break;
                }
                case "MakeNamedMedium": this.nextString(tokens, c); this.readParams(tokens, c); break;
                case "MediumInterface": this.nextString(tokens, c); if (tokens[c.i]?.kind === "string") this.nextString(tokens, c); break;
                // --- Lights ---
                case "AreaLightSource": {
                    const type = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    if (type !== "diffuse") this.warn(`unsupported area light '${type}'`);
                    this.state.areaLight = params;
                    break;
                }
                case "LightSource": {
                    const type = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    this.addLight(type, params);
                    break;
                }
                // --- Shapes ---
                case "Shape": {
                    const type = this.nextString(tokens, c);
                    const params = this.readParams(tokens, c);
                    await this.addShape(type, params);
                    break;
                }
                // --- Includes ---
                case "Include": case "Import": {
                    const file = this.nextString(tokens, c);
                    const text = await this.fetchText(file);
                    await this.run(tokenize(text));
                    break;
                }
                default:
                    this.warn(`ignoring unknown directive '${d}'`);
                    break;
            }
        }
    }

    private readMatrix(tokens: Token[], c: { i: number }): float4x4 {
        const v = this.nextFloats(tokens, c, 16);
        // pbrt matrices are column-major; float4x4 is row-major -> transpose on read.
        const m = float4x4.identity();
        for (let col = 0; col < 4; col++) for (let row = 0; row < 4; row++) m.set(row, col, v[col * 4 + row]!);
        return m;
    }

    private async fetchText(rel: string): Promise<string> {
        const url = await resolveAssetUrl(rel, this.baseUrl, AssetCategory.Scene);
        const res = await fetch(url);
        if (!res.ok) throw new RuntimeError(`pbrt: failed to fetch '${url}' (${res.status})`);
        return res.text();
    }

    // ---- Camera ----

    private applyCamera(cameraFromWorld: float4x4, params: Params): void {
        const fov = P.float(params, "fov", 90);
        const node = mulMat(inverse(cameraFromWorld), kInvertZ); // worldFromCamera * invertZ
        const cam = new CameraBridge();
        cam.position = transformPoint(node, new float3(0, 0, 0));
        cam.target = transformPoint(node, new float3(0, 0, -1)); // Falcor looks down -z
        cam.up = normalize3(transformVector(node, new float3(0, 1, 0)));
        cam.focalLength = fovYToFocalLength(radians(fov), 24);
        this.builder.addCamera(cam);
    }

    // ---- Lights ----

    private addLight(type: string, params: Params): void {
        if (type === "distant") {
            // Native defaults L to the D65 illuminant (unit luminance).
            const L = P.rgb(params, "L", namedSpectrumRGB("stdillum-D65"));
            const scale = P.float(params, "scale", 1);
            const from = P.has(params, "from") ? P.rgb(params, "from", new float3(0, 0, 0)) : new float3(0, 0, 0);
            const to = P.has(params, "to") ? P.rgb(params, "to", new float3(0, 0, 1)) : new float3(0, 0, 1);
            const dir = normalize3(transformVector(this.state.ctm, sub3(to, from)));
            const light = new LightBridge(LightType.Directional, "DirectionalLight");
            light.intensity = new float3(L.x * scale, L.y * scale, L.z * scale);
            light.direction = dir;
            this.builder.addLight(light);
        } else if (type === "infinite") {
            const filename = P.string(params, "filename", "");
            const scale = P.float(params, "scale", 1);
            const hasL = P.has(params, "L");
            if (hasL && filename) throw new RuntimeError("pbrt: can't specify both emission 'L' and 'filename' for infinite light");
            if (hasL) {
                // Falcor has no constant environment emitter: native builds a
                // one-pixel env map (and leaves `scale` unapplied there).
                const L = P.rgb(params, "L", new float3(1, 1, 1));
                this.builder.envMap = { path: "", intensity: 1, constantColor: [L.x, L.y, L.z] };
            } else if (filename) {
                // pbrt-v4 stores env maps equal-area octahedral; native converts
                // them to lat-long and orients them by the light's transform.
                const rotation = extractEulerAngleXYZ(this.state.ctm);
                const degrees = (r: number) => (r * 180) / Math.PI;
                this.builder.envMap = {
                    path: filename,
                    intensity: scale,
                    equalAreaOctahedral: true,
                    rotation: { x: degrees(rotation.x), y: degrees(rotation.y), z: degrees(rotation.z) },
                };
            }
        } else {
            this.warn(`unsupported light type '${type}'`);
        }
    }

    // ---- Shapes ----

    private async addShape(type: string, params: Params): Promise<void> {
        let mesh: TriangleMeshDesc | null = null;
        switch (type) {
            case "trianglemesh": {
                const P3 = P.floats(params, "P");
                let indices = P.ints(params, "indices");
                const positions: float3[] = [];
                for (let i = 0; i + 2 < P3.length; i += 3) positions.push(new float3(P3[i]!, P3[i + 1]!, P3[i + 2]!));
                if (indices.length === 0 && positions.length === 3) indices = [0, 1, 2];
                while (indices.length % 3 !== 0) indices.pop();
                if (positions.length === 0 || indices.length === 0) { this.warn("trianglemesh missing P or indices"); return; }
                const nFlat = P.floats(params, "N");
                const normals: float3[] = [];
                if (nFlat.length === positions.length * 3) for (let i = 0; i < nFlat.length; i += 3) normals.push(new float3(nFlat[i]!, nFlat[i + 1]!, nFlat[i + 2]!));
                const uvFlat = P.floats(params, "uv");
                const uvs: float2[] = [];
                if (uvFlat.length === positions.length * 2) for (let i = 0; i < uvFlat.length; i += 2) uvs.push(new float2(uvFlat[i]!, uvFlat[i + 1]!));
                mesh = assembleMesh(positions, indices, normals, uvs);
                break;
            }
            case "plymesh": {
                const file = P.string(params, "filename", "");
                if (!file) { this.warn("plymesh missing filename"); return; }
                const url = await resolveAssetUrl(file, this.baseUrl);
                const res = await fetch(url);
                if (!res.ok) { this.warn(`plymesh fetch failed '${url}' (${res.status})`); return; }
                mesh = parsePly(await res.arrayBuffer());
                break;
            }
            case "sphere": {
                mesh = TriangleMesh.createSphere(P.float(params, "radius", 1));
                break;
            }
            case "disk": {
                // pbrt disk lies in the XY plane at z=height (Falcor uses XZ + kYtoZ).
                mesh = this.makeDisk(P.float(params, "radius", 1), P.float(params, "height", 0));
                break;
            }
            default:
                this.warn(`unsupported shape '${type}'`);
                return;
        }
        if (!mesh) return;

        const materialDef = this.state.material ?? { type: "diffuse", name: "", params: new Map() };
        let mb: MaterialBridge;
        if (this.state.areaLight) {
            mb = this.translateMaterial(materialDef, true);
            const L = P.rgb(this.state.areaLight, "L", new float3(1, 1, 1));
            mb.emissiveColor = L;
            // Native reads `scale` for diffuse area lights but never applies it.
            mb.emissiveFactor = 1;
        } else {
            const cached = this.matCache.get(materialDef);
            if (cached) mb = cached;
            else { mb = this.translateMaterial(materialDef); this.matCache.set(materialDef, mb); }
        }

        const meshID = this.builder.addTriangleMesh(mesh, mb);
        const nodeID = this.builder.addNode(type, this.state.ctm.clone());
        this.builder.addMeshInstance(nodeID, meshID);
    }

    private makeDisk(radius: number, height: number): TriangleMeshDesc {
        const segs = 64;
        const positions: float3[] = [new float3(0, 0, height)];
        for (let i = 0; i < segs; i++) {
            const a = (i / segs) * 2 * Math.PI;
            positions.push(new float3(radius * Math.cos(a), radius * Math.sin(a), height));
        }
        const indices: number[] = [];
        for (let i = 1; i <= segs; i++) indices.push(0, i, (i % segs) + 1);
        return assembleMesh(positions, indices, [], []);
    }

    /**
     * Mirrors getSpectrumTexture: a material parameter may name a `Texture`
     * instead of holding a constant. Image textures are bound to the material
     * slot; the constant ones fold into a value (native does the same).
     *
     * @returns The constant to use, plus the image file when one should be bound.
     */
    private resolveTextureParam(p: Params, name: string, def: float3): { value: float3; filename?: string } {
        const v = p.get(name);
        if (!v || v.type !== "texture" || !v.values.length) return { value: P.rgb(p, name, def) };

        const seen = new Set<string>();
        let scale = new float3(1, 1, 1);
        let texture = this.textures.get(v.values[0]!);
        // `scale` textures wrap another texture with a multiplier; follow the chain.
        while (texture && texture.textureClass === "scale") {
            const factor = P.rgb(texture.params, "scale", new float3(1, 1, 1));
            scale = new float3(scale.x * factor.x, scale.y * factor.y, scale.z * factor.z);
            const inner = texture.params.get("tex");
            if (!inner || inner.type !== "texture" || !inner.values.length) {
                const constant = P.rgb(texture.params, "tex", def);
                return { value: new float3(constant.x * scale.x, constant.y * scale.y, constant.z * scale.z) };
            }
            if (seen.has(inner.values[0]!)) break; // guard against cycles
            seen.add(inner.values[0]!);
            texture = this.textures.get(inner.values[0]!);
        }
        if (!texture) {
            this.warn(`cannot find texture named '${v.values[0]}' for parameter '${name}'`);
            return { value: def };
        }
        if (texture.textureClass === "constant") {
            const constant = P.rgb(texture.params, "value", new float3(1, 1, 1));
            return { value: new float3(constant.x * scale.x, constant.y * scale.y, constant.z * scale.z) };
        }
        if (texture.textureClass === "imagemap") {
            const filename = P.string(texture.params, "filename", "");
            if (!filename) {
                this.warn(`imagemap texture '${v.values[0]}' has no filename`);
                return { value: def };
            }
            // The image modulates the constant; pbrt's own scale parameter folds in too.
            const imageScale = P.float(texture.params, "scale", 1);
            return { value: new float3(scale.x * imageScale, scale.y * imageScale, scale.z * imageScale), filename };
        }
        // checkerboard/mix/noise textures have no Falcor equivalent (native warns too).
        this.warn(`pbrt texture class '${texture.textureClass}' is not supported (parameter '${name}')`);
        return { value: def };
    }

    private translateMaterial(def: MaterialDef, isAreaLight = false): MaterialBridge {
        const p = def.params;
        const normalMap = P.string(p, "normalmap", "");
        // Mirrors the native Settings option; area-light materials always take
        // the StandardMaterial path (as in native createMaterial).
        const usePBRT = !isAreaLight && getGlobalSettings().getOption<boolean>("PBRTImporter:usePBRTMaterials", false) === true;
        const m = new MaterialBridge(MaterialType.Standard, def.name);
        m.doubleSided = true;
        switch (def.type) {
            case "":
            case "none":
            case "interface":
            case "diffuse": {
                const reflectance = this.resolveTextureParam(p, "reflectance", new float3(0.5, 0.5, 0.5));
                const refl = reflectance.value;
                if (reflectance.filename) {
                    m.loadTexture("BaseColor", reflectance.filename);
                    if (usePBRT) this.warn("textured reflectance falls back to StandardMaterial (PBRT material classes take constants)");
                }
                if (usePBRT && def.type === "diffuse" && !reflectance.filename) {
                    const pm = new MaterialBridge(MaterialType.PBRTDiffuse, def.name);
                    pm.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                    pm.doubleSided = true;
                    return pm;
                }
                m.metallic = 0;
                m.roughness = 1;
                m.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                break;
            }
            case "coateddiffuse": {
                const coated = this.resolveTextureParam(p, "reflectance", new float3(0.5, 0.5, 0.5));
                const refl = coated.value;
                if (coated.filename) m.loadTexture("BaseColor", coated.filename);
                if (usePBRT && !coated.filename) {
                    const a = scalarRoughnessAlpha(p);
                    const pm = new MaterialBridge(MaterialType.PBRTCoatedDiffuse, def.name);
                    pm.roughness = { x: a, y: a }; // setRoughness(float2) -> specular.rg
                    pm.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                    pm.doubleSided = true;
                    return pm;
                }
                m.metallic = 0;
                m.roughness = Math.sqrt(scalarRoughnessAlpha(p));
                m.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                break;
            }
            case "conductor": {
                if (usePBRT) {
                    const { eta, k } = conductorEtaK(p);
                    const a = scalarRoughnessAlpha(p);
                    const pm = new MaterialBridge(MaterialType.PBRTConductor, def.name);
                    pm.baseColor = new float4(eta.x, eta.y, eta.z, 1);
                    pm.transmissionColor = k;
                    pm.roughness = { x: a, y: a };
                    pm.doubleSided = true;
                    return pm;
                }
                const albedo = conductorSpecularAlbedo(p);
                m.baseColor = new float4(albedo.x, albedo.y, albedo.z, 1);
                m.metallic = 1;
                m.roughness = Math.sqrt(scalarRoughnessAlpha(p));
                break;
            }
            case "coatedconductor": {
                // Native uses the conductor.*-prefixed parameter names for this type.
                if (usePBRT) {
                    const ia = scalarRoughnessAlpha(p, "interface.roughness", "interface.uroughness", "interface.vroughness");
                    const ca = scalarRoughnessAlpha(p, "conductor.roughness", "conductor.uroughness", "conductor.vroughness");
                    const { eta, k } = conductorEtaK(p, "conductor.reflectance", "conductor.eta", "conductor.k");
                    const pm = new MaterialBridge(MaterialType.PBRTCoatedConductor, def.name);
                    pm.baseColor = new float4(eta.x, eta.y, eta.z, 1);
                    pm.transmissionColor = k;
                    pm.specularParams = new float4(ia, ia, ca, ca); // (interfaceRoughness, conductorRoughness)
                    pm.indexOfRefraction = scalarEta(p, "interface.eta");
                    pm.doubleSided = true;
                    return pm;
                }
                const albedo = conductorSpecularAlbedo(p, "conductor.reflectance", "conductor.eta", "conductor.k");
                m.baseColor = new float4(albedo.x, albedo.y, albedo.z, 1);
                m.metallic = 1;
                m.roughness = Math.sqrt(scalarRoughnessAlpha(p, "conductor.roughness", "conductor.uroughness", "conductor.vroughness"));
                break;
            }
            case "dielectric": {
                if (usePBRT) {
                    const a = scalarRoughnessAlpha(p);
                    const pm = new MaterialBridge(MaterialType.PBRTDielectric, def.name);
                    pm.roughness = { x: a, y: a };
                    pm.indexOfRefraction = scalarEta(p);
                    return pm; // native leaves doubleSided false here
                }
                m.metallic = 0;
                m.roughness = Math.sqrt(scalarRoughnessAlpha(p));
                m.indexOfRefraction = scalarEta(p);
                m.specularTransmission = 1;
                break;
            }
            case "thindielectric": {
                // No PBRT-material branch natively either.
                m.metallic = 0;
                m.roughness = 0;
                m.indexOfRefraction = scalarEta(p);
                m.specularTransmission = 1;
                m.thinSurface = true;
                break;
            }
            case "diffusetransmission": {
                const refl = P.rgb(p, "reflectance", new float3(0.25, 0.25, 0.25));
                const trans = P.rgb(p, "transmittance", new float3(0.25, 0.25, 0.25));
                if (usePBRT) {
                    const pm = new MaterialBridge(MaterialType.PBRTDiffuseTransmission, def.name);
                    pm.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                    pm.transmissionColor = trans;
                    pm.doubleSided = true;
                    return pm;
                }
                m.metallic = 0;
                m.roughness = 1;
                m.diffuseTransmission = 0.5;
                m.baseColor = new float4(refl.x, refl.y, refl.z, 1);
                m.transmissionColor = trans;
                break;
            }
            default: {
                this.warn(`unsupported material '${def.type}' -> diffuse`);
                m.metallic = 0;
                m.roughness = 1;
                m.baseColor = new float4(0.5, 0.5, 0.5, 1);
                break;
            }
        }
        // Mirrors the native importer binding "normalmap" to the Normal slot.
        if (normalMap) m.loadTexture("Normal", normalMap);
        return m;
    }
}

/** Left-handed lookAt matrix (pbrt `LookAt`, mirrors math::matrixFromLookAt LH). */
function lookAtLH(eye: float3, center: float3, up: float3): float4x4 {
    const f = normalize3(sub3(center, eye));
    const r = normalize3(cross(up, f));
    const u = cross(f, r);
    const m = float4x4.identity();
    m.set(0, 0, r.x); m.set(0, 1, r.y); m.set(0, 2, r.z);
    m.set(1, 0, u.x); m.set(1, 1, u.y); m.set(1, 2, u.z);
    m.set(2, 0, f.x); m.set(2, 1, f.y); m.set(2, 2, f.z);
    m.set(0, 3, -dot3(r, eye));
    m.set(1, 3, -dot3(u, eye));
    m.set(2, 3, -dot3(f, eye));
    return m;
}

/**
 * Parses a pbrt-v4 scene file and builds a Scene (assets fetched relative to
 * baseUrl). Parallels runSceneScript for .pyscene.
 */
export async function runPbrtScene(device: Device, source: string, baseUrl: string): Promise<Scene> {
    return withScriptSearchPath(baseUrl, () => new PbrtScene(device, baseUrl).load(source));
}
