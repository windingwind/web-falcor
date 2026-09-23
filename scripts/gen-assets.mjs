#!/usr/bin/env node
/**
 * Asset generator — writes format-exact files for the asset types that cannot
 * simply be downloaded, into `Falcor/media/` next to the fetched ones.
 *
 * Why generate:
 *   • MERL BRDF database — free for research, but only behind MERL's licence
 *     agreement, so it cannot be fetched reproducibly. The `.binary` format is
 *     public (90x90x180 bins of RGB doubles), so the loader is exercised with
 *     analytic BRDFs written in exactly that format. Native Falcor loads these
 *     files too, so they work as oracles on both sides.
 *   • IES photometry — manufacturers publish `.ies` files freely but under no
 *     reusable licence. The format (IESNA LM-63) is a published text format, so
 *     the profiles here are written to spec with analytic candela distributions.
 *
 * The analytic BRDFs are chosen to be *checkable*:
 *   • `merl-lambert.binary` is a constant (Lambertian) BRDF with a known albedo,
 *     so both the evaluation and the integrated albedo LUT have closed forms.
 *   • `merl-index-probe.binary` stores each bin's own (thetaH, thetaD, phiD)
 *     index, normalized — any slip in the half/difference-vector mapping or the
 *     buffer layout shows up directly as a wrong readback.
 *   • `merl-lambert-green.binary` and `merl-lambert-blue.binary` plus
 *     `merl-index-map.tga` give MERLMix three constant BRDFs and a selector
 *     whose every texel is known, so the per-texel choice — including the
 *     wrap-around modulo a non-power-of-two BRDF count — is exactly checkable.
 *
 * Usage:
 *   node scripts/gen-assets.mjs              # everything (~135 MB)
 *   node scripts/gen-assets.mjs --list       # show what would be written
 *   node scripts/gen-assets.mjs merl         # only the named group(s)
 *   node scripts/gen-assets.mjs --force      # rewrite files that already exist
 *   node scripts/gen-assets.mjs --dest dir   # write elsewhere
 */

import { writeFileSync, mkdirSync, existsSync, statSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// MERL BRDF database layout (Matusik et al. 2003), as read by Falcor's MERLFile.
const kThetaH = 90;
const kThetaD = 90;
const kPhiD = 360;
const kSampleCount = (kThetaH * kThetaD * kPhiD) / 2; // 1,458,000 bins
// Values are stored pre-divided by these per-channel scales.
const kScale = [1.0 / 1500.0, 1.15 / 1500.0, 1.66 / 1500.0];

/**
 * Writes a MERL `.binary`: three int32 dimensions followed by the R, G and B
 * planes of `kSampleCount` doubles each.
 *
 * @param brdf (thetaHIndex, thetaDIndex, phiDIndex) -> [r, g, b] BRDF values.
 */
function writeMerlBinary(path, brdf) {
    const header = Buffer.alloc(12);
    header.writeInt32LE(kThetaH, 0);
    header.writeInt32LE(kThetaD, 4);
    header.writeInt32LE(kPhiD / 2, 8);

    const planes = Buffer.alloc(kSampleCount * 3 * 8);
    const rgb = [0, 0, 0];
    for (let h = 0; h < kThetaH; h++) {
        for (let d = 0; d < kThetaD; d++) {
            for (let p = 0; p < kPhiD / 2; p++) {
                // Falcor: idx = (thetaD + thetaH * kThetaD) * (kPhiD / 2) + phiD
                const idx = (d + h * kThetaD) * (kPhiD / 2) + p;
                brdf(h, d, p, rgb);
                for (let c = 0; c < 3; c++) planes.writeDoubleLE(rgb[c] / kScale[c], (c * kSampleCount + idx) * 8);
            }
        }
    }
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, Buffer.concat([header, planes]));
    return header.length + planes.length;
}

/**
 * Writes an IESNA LM-63-2002 photometric file.
 *
 * Layout after the keyword block: TILT=NONE, then
 *   <lamps> <lumens> <multiplier> <#vertical> <#horizontal> <photometric type>
 *   <units> <width> <length> <height>
 *   <ballast factor> <ballast lamp factor> <input watts>
 *   <vertical angles> <horizontal angles> <candela values>
 * with the candela block ordered by horizontal angle, then vertical.
 *
 * @param candela (verticalDeg, horizontalDeg) -> candela.
 */
function writeIesProfile(path, verticalAngles, horizontalAngles, candela) {
    const lines = [
        "IESNA:LM-63-2002",
        "[TEST] web-falcor synthetic profile",
        "[MANUFAC] web-falcor scripts/gen-assets.mjs",
        "TILT=NONE",
        `1 1000 1 ${verticalAngles.length} ${horizontalAngles.length} 1 1 0 0 0`,
        "1 1 100",
    ];
    // The parser is whitespace-driven; wrap long rows for readability like real files.
    const wrap = (values) => {
        const out = [];
        for (let i = 0; i < values.length; i += 10) out.push(values.slice(i, i + 10).map((v) => v.toFixed(4)).join(" "));
        return out;
    };
    lines.push(...wrap(verticalAngles));
    lines.push(...wrap(horizontalAngles));
    for (const h of horizontalAngles) lines.push(...wrap(verticalAngles.map((v) => candela(v, h))));

    const text = lines.join("\n") + "\n";
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, text);
    return text.length;
}

/**
 * Writes an SDF grid value file (`.sdfg`): a uint32 grid width followed by
 * (width + 1)^3 float corner values. The grid occupies the unit cube, so corner
 * (x, y, z) sits at `xyz / gridWidth - 0.5`, and native clamps the signed
 * distances to +/- sqrt(3) (the cube's diagonal).
 */
function writeSdfGrid(path, gridWidth, signedDistance) {
    const widthInValues = gridWidth + 1;
    const total = widthInValues * widthInValues * widthInValues;
    const out = Buffer.alloc(4 + total * 4);
    out.writeUInt32LE(gridWidth, 0);
    const limit = Math.sqrt(3);
    for (let z = 0; z < widthInValues; z++) {
        for (let y = 0; y < widthInValues; y++) {
            for (let x = 0; x < widthInValues; x++) {
                const p = [x / gridWidth - 0.5, y / gridWidth - 0.5, z / gridWidth - 0.5];
                const sd = Math.min(Math.max(signedDistance(p), -limit), limit);
                out.writeFloatLE(sd, 4 + (x + widthInValues * (y + widthInValues * z)) * 4);
            }
        }
    }
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, out);
    return out.length;
}

/**
 * Writes an uncompressed 8-bit grayscale TGA with a top-left origin. MERLMix
 * index maps are exactly this: one BRDF index per texel in the red channel.
 *
 * @param value (x, y) -> index byte.
 */
function writeGrayTga(path, width, height, value) {
    const header = Buffer.alloc(18);
    header[2] = 3; // uncompressed grayscale
    header.writeUInt16LE(width, 12);
    header.writeUInt16LE(height, 14);
    header[16] = 8; // bits per pixel
    header[17] = 0x20; // origin in the top-left corner
    const pixels = Buffer.alloc(width * height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) pixels[y * width + x] = value(x, y) & 0xff;
    }
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, Buffer.concat([header, pixels]));
    return header.length + pixels.length;
}

/** One SDF3DPrimitive in the `.sdf` JSON layout (SDFGrid.cpp's to_json keys). */
function prim(shapeType, shapeData, operationType, opts = {}) {
    return {
        shape_type: shapeType,
        shape_data: shapeData,
        shape_blobbing: opts.blobbing ?? 0,
        operation_type: operationType,
        operation_smoothing: opts.smoothing ?? 0,
        translation: opts.translation ?? [0, 0, 0],
        inv_rot_scale: opts.invRotScale ?? [1, 0, 0, 0, 1, 0, 0, 0, 1],
    };
}

/** Writes a `.sdf` primitive list (SDFGrid::writePrimitivesToFile's 4-space JSON). */
function writeSdfPrimitives(path, primitives) {
    const text = JSON.stringify(primitives, null, 4);
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, text);
    return text.length;
}

const degToRad = (d) => (d * Math.PI) / 180;
const range = (n, f) => Array.from({ length: n }, (_v, i) => f(i));

const GROUPS = [
    {
        name: "merl",
        dir: "merl",
        note: "Analytic BRDFs in the MERL .binary format (the real database is licence-gated)",
        files: [
            {
                file: "merl-lambert.binary",
                sizeMB: 34,
                desc: "Lambertian, albedo (0.6, 0.5, 0.4): f = albedo / pi in every bin",
                write: (path) =>
                    writeMerlBinary(path, (_h, _d, _p, out) => {
                        out[0] = 0.6 / Math.PI;
                        out[1] = 0.5 / Math.PI;
                        out[2] = 0.4 / Math.PI;
                    }),
            },
            {
                file: "merl-lambert-green.binary",
                sizeMB: 34,
                desc: "Lambertian, albedo (0.2, 0.7, 0.3): a second constant BRDF for MERLMix",
                write: (path) =>
                    writeMerlBinary(path, (_h, _d, _p, out) => {
                        out[0] = 0.2 / Math.PI;
                        out[1] = 0.7 / Math.PI;
                        out[2] = 0.3 / Math.PI;
                    }),
            },
            {
                file: "merl-lambert-blue.binary",
                sizeMB: 34,
                desc: "Lambertian, albedo (0.1, 0.2, 0.9): a third constant BRDF for MERLMix",
                write: (path) =>
                    writeMerlBinary(path, (_h, _d, _p, out) => {
                        out[0] = 0.1 / Math.PI;
                        out[1] = 0.2 / Math.PI;
                        out[2] = 0.9 / Math.PI;
                    }),
            },
            {
                file: "merl-index-map.tga",
                sizeMB: 1,
                desc: "8x8 MERLMix selector: texel (x, y) holds the index x + 8y (0..63)",
                write: (path) => writeGrayTga(path, 8, 8, (x, y) => x + 8 * y),
            },
            {
                file: "merl-index-probe.binary",
                sizeMB: 34,
                desc: "Each bin stores its own normalized (thetaH, thetaD, phiD) index",
                write: (path) =>
                    writeMerlBinary(path, (h, d, p, out) => {
                        out[0] = h / (kThetaH - 1);
                        out[1] = d / (kThetaD - 1);
                        out[2] = p / (kPhiD / 2 - 1);
                    }),
            },
        ],
    },
    {
        name: "ies",
        dir: "ies",
        note: "IESNA LM-63 photometric profiles with analytic candela distributions",
        files: [
            {
                file: "ies-cosine.ies",
                sizeMB: 1,
                desc: "Rotationally symmetric cosine lobe: I(v) = 1000 * cos(v), 0..90 deg",
                write: (path) =>
                    writeIesProfile(
                        path,
                        range(19, (i) => i * 5), // 0..90 in 5 degree steps
                        [0],
                        (v) => 1000 * Math.cos(degToRad(v)),
                    ),
            },
            {
                file: "ies-spot.ies",
                sizeMB: 1,
                desc: "Narrow symmetric spot: I(v) = 2000 * cos(v)^16 out to 45 deg",
                write: (path) =>
                    writeIesProfile(
                        path,
                        range(46, (i) => i), // 0..45 in 1 degree steps
                        [0],
                        (v) => 2000 * Math.pow(Math.cos(degToRad(v)), 16),
                    ),
            },
            {
                file: "ies-asymmetric.ies",
                sizeMB: 1,
                desc: "Azimuth-dependent profile over 0..180 deg horizontal (quadrilateral symmetry)",
                write: (path) =>
                    writeIesProfile(
                        path,
                        range(19, (i) => i * 5),
                        range(7, (i) => i * 30), // 0..180 in 30 degree steps
                        (v, h) => 1000 * Math.cos(degToRad(v)) * (1 + 0.5 * Math.cos(degToRad(h))),
                    ),
            },
        ],
    },
    {
        name: "sdf",
        dir: "sdf",
        note: "SDF grid corner values (.sdfg) and primitive lists (.sdf) with analytic signed distance fields",
        files: [
            {
                file: "sdf-sphere-64.sdfg",
                sizeMB: 1,
                desc: "Sphere of radius 0.4 centred in the unit cube, 64^3 cells",
                write: (path) => writeSdfGrid(path, 64, (p) => Math.hypot(p[0], p[1], p[2]) - 0.4),
            },
            {
                file: "sdf-primitives.sdf",
                sizeMB: 1,
                desc: "CSG primitive list: sphere ∪ 45°-rotated bar, minus a sphere at the top",
                write: (path) => {
                    const c = Math.SQRT1_2; // cos(45°) = sin(45°)
                    return writeSdfPrimitives(path, [
                        prim("sphere", [0.28, 0, 0], "union"),
                        // Rotated about z, so a dropped transpose in the shader
                        // would put the bar on the other diagonal.
                        prim("box", [0.42, 0.08, 0.08], "union", { invRotScale: [c, -c, 0, c, c, 0, 0, 0, 1] }),
                        prim("sphere", [0.12, 0, 0], "subtraction", { translation: [0, 0, 0.22] }),
                    ]);
                },
            },
            {
                file: "sdf-box-64.sdfg",
                sizeMB: 1,
                desc: "Box of half-extent 0.3, 64^3 cells",
                write: (path) =>
                    writeSdfGrid(path, 64, (p) => {
                        const d = p.map((v) => Math.abs(v) - 0.3);
                        const outside = Math.hypot(Math.max(d[0], 0), Math.max(d[1], 0), Math.max(d[2], 0));
                        return outside + Math.min(Math.max(d[0], d[1], d[2]), 0);
                    }),
            },
        ],
    },
];

function parseArgs(argv) {
    const opts = { list: false, force: false, help: false, dest: join(repoRoot, "Falcor/media"), names: [] };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--list" || a === "-l") opts.list = true;
        else if (a === "--force" || a === "-f") opts.force = true;
        else if (a === "--help" || a === "-h") opts.help = true;
        else if (a === "--dest") opts.dest = resolve(argv[++i] ?? "");
        else if (a.startsWith("--dest=")) opts.dest = resolve(a.slice("--dest=".length));
        else if (a.startsWith("-")) throw new Error(`unknown option "${a}"`);
        else opts.names.push(a.toLowerCase());
    }
    return opts;
}

function main() {
    const opts = parseArgs(process.argv.slice(2));
    if (opts.help) {
        console.log(`Generate format-exact assets into Falcor/media/.

  node scripts/gen-assets.mjs [groups...] [--list] [--force] [--dest <dir>]

Groups: ${GROUPS.map((g) => g.name).join(", ")}`);
        return;
    }
    if (opts.list) {
        for (const g of GROUPS) {
            console.log(`\n${g.name}  →  ${g.dir}/`);
            console.log(`  ${g.note}`);
            for (const f of g.files) console.log(`    ${f.file.padEnd(28)} ~${String(f.sizeMB).padStart(3)} MB  ${f.desc}`);
        }
        return;
    }

    const wanted = opts.names.length
        ? opts.names.map((n) => {
              const g = GROUPS.find((x) => x.name === n);
              if (!g) throw new Error(`unknown group "${n}" (known: ${GROUPS.map((x) => x.name).join(", ")})`);
              return g;
          })
        : GROUPS;

    for (const group of wanted) {
        const dir = join(opts.dest, group.dir);
        console.log(`\n${group.name} → ${dir}`);
        for (const f of group.files) {
            const path = join(dir, f.file);
            if (!opts.force && existsSync(path) && statSync(path).size > 0) {
                console.log(`  ${f.file} — already present (${(statSync(path).size / 1048576).toFixed(1)} MB)`);
                continue;
            }
            process.stdout.write(`  ${f.file} … `);
            const start = Date.now();
            const bytes = f.write(path);
            console.log(`${(bytes / 1048576).toFixed(1)} MB in ${((Date.now() - start) / 1000).toFixed(1)}s`);
        }
    }
}

try {
    main();
} catch (err) {
    console.error(`error: ${err.message}`);
    process.exit(1);
}
