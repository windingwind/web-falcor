#!/usr/bin/env node
/**
 * Asset downloader — provisions the *non-scene* assets Falcor features need but
 * that ship with neither the Falcor repo nor its media bundle: measured-BRDF
 * data, compressed OpenVDB volumes, IES light profiles.
 *
 * `download:scenes` covers whole scenes (Falcor's media bundle, ORCA, pbrt);
 * this script covers the single-file asset formats the loaders consume. Both
 * write into `Falcor/media/` (gitignored, served by the dev server and the GPU
 * test harness at `/Falcor/media/...`).
 *
 * Usage:
 *   node scripts/download-assets.mjs                  # the default set (small, ~10 MB)
 *   node scripts/download-assets.mjs openvdb rgl      # only the named groups
 *   node scripts/download-assets.mjs --list           # show the catalog and exit
 *   node scripts/download-assets.mjs --all            # everything, incl. the large volumes
 *   node scripts/download-assets.mjs --force openvdb  # re-download even if present
 *   node scripts/download-assets.mjs --dest some/dir  # write elsewhere
 *
 * (via npm: `npm run download:assets -- rgl`)
 *
 * Provenance / licensing (see also each group's `note`):
 *   • OpenVDB sample models — published by the Academy Software Foundation on
 *     openvdb.org for testing; source meshes from NASA 3D Resources and the
 *     Stanford 3D Scanning Repository.
 *   • RGL material database (Dupuy & Jakob 2018) — CC0 1.0 public domain.
 *
 * Formats whose reference data is license-gated (MERL) or not published under a
 * reusable license (manufacturer IES photometry) are *generated* instead, by
 * `tools/gen-assets.mjs` — format-exact files that native Falcor loads too, so
 * they still serve as oracles.
 */

import { createWriteStream, mkdirSync, existsSync, statSync, rmSync, renameSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const OPENVDB_BASE = "https://media.githubusercontent.com/media/AcademySoftwareFoundation/openvdb-website/master/download/models";
const RGL_BASE = "https://d38rqfq1h7iukm.cloudfront.net/media/materials";

/**
 * Catalog. Each group lands in `<dest>/<group.dir>/`; `files` are downloaded
 * verbatim. `small` groups are the default set (kept under ~10 MB total so the
 * GPU suite can run on a fresh clone without the big volumes).
 */
const GROUPS = [
    {
        name: "openvdb",
        dir: "openvdb",
        small: true,
        note: "OpenVDB sample models (blosc-compressed, half-float level sets / fog volumes) — openvdb.org test data",
        files: [
            { file: "cube.vdb", url: `${OPENVDB_BASE}/cube.vdb`, sizeMB: 2, desc: "Level-set cube (blosc + active mask, UniformScaleMap)" },
            { file: "sphere.vdb", url: `${OPENVDB_BASE}/sphere.vdb`, sizeMB: 2, desc: "Level-set sphere" },
            { file: "torus.vdb", url: `${OPENVDB_BASE}/torus.vdb`, sizeMB: 2, desc: "Level-set torus" },
            { file: "smoke.vdb", url: `${OPENVDB_BASE}/smoke.vdb`, sizeMB: 2, desc: "Fog volume (density grid)" },
        ],
    },
    {
        name: "openvdb-large",
        dir: "openvdb",
        note: "Larger OpenVDB sample models (tens of MB) — same source as `openvdb`",
        files: [
            { file: "bunny_cloud.vdb", url: `${OPENVDB_BASE}/bunny_cloud.vdb`, sizeMB: 73, desc: "Stanford bunny as a fog volume" },
            { file: "explosion.vdb", url: `${OPENVDB_BASE}/explosion.vdb`, sizeMB: 25, desc: "Explosion (density + temperature grids)" },
            { file: "dragon.vdb", url: `${OPENVDB_BASE}/dragon.vdb`, sizeMB: 25, desc: "Stanford dragon level set" },
        ],
    },
    {
        name: "rgl",
        dir: "rgl",
        small: true,
        note: "RGL measured BSDFs (Dupuy & Jakob 2018, rgl.epfl.ch/materials) — CC0 1.0; feed RGLMaterial",
        files: [
            { file: "acrylic_felt_green_rgb.bsdf", url: `${RGL_BASE}/acrylic_felt_green/acrylic_felt_green_rgb.bsdf`, sizeMB: 1, desc: "Green acrylic felt (isotropic, RGB)" },
            { file: "chm_mint_rgb.bsdf", url: `${RGL_BASE}/chm_mint/chm_mint_rgb.bsdf`, sizeMB: 1, desc: "Mint car paint (isotropic, RGB)" },
            { file: "aniso_morpho_melenaus_rgb.bsdf", url: `${RGL_BASE}/aniso_morpho_melenaus/aniso_morpho_melenaus_rgb.bsdf`, sizeMB: 4, desc: "Morpho butterfly wing (anisotropic, RGB)" },
        ],
    },
];

const bySmall = () => GROUPS.filter((g) => g.small);

function parseArgs(argv) {
    const opts = { list: false, force: false, all: false, help: false, dest: join(repoRoot, "Falcor/media"), names: [] };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--list" || a === "-l") opts.list = true;
        else if (a === "--force" || a === "-f") opts.force = true;
        else if (a === "--all") opts.all = true;
        else if (a === "--help" || a === "-h") opts.help = true;
        else if (a === "--dest") opts.dest = resolve(argv[++i] ?? "");
        else if (a.startsWith("--dest=")) opts.dest = resolve(a.slice("--dest=".length));
        else if (a.startsWith("-")) throw new Error(`unknown option "${a}"`);
        else opts.names.push(a);
    }
    return opts;
}

function usage() {
    console.log(`Download the single-file assets Falcor loaders need into Falcor/media/.

  node scripts/download-assets.mjs [groups...] [options]

Options:
      --list        List the catalog and exit
      --all         Download every group (incl. the large volumes)
      --force       Re-download files that are already present
      --dest <dir>  Write into <dir> (default: Falcor/media)

Groups: ${GROUPS.map((g) => g.name).join(", ")}
Default (no arguments): ${bySmall().map((g) => g.name).join(", ")}`);
}

function list() {
    for (const g of GROUPS) {
        const total = g.files.reduce((s, f) => s + f.sizeMB, 0);
        console.log(`\n${g.name}${g.small ? " (default)" : ""}  →  ${g.dir}/   ~${total} MB`);
        console.log(`  ${g.note}`);
        for (const f of g.files) console.log(`    ${f.file.padEnd(34)} ${String(f.sizeMB).padStart(3)} MB  ${f.desc}`);
    }
    console.log("\nAssets keep their upstream licenses; see the header of this script for provenance.");
}

/** Stream a URL to `dest` with a progress bar and retry-with-backoff. */
async function downloadTo(url, dest, label, attempts = 4) {
    for (let a = 1; ; a++) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const total = Number(res.headers.get("content-length")) || 0;
            const tmp = `${dest}.part`;
            const out = createWriteStream(tmp);
            let received = 0;
            let lastPct = -1;
            const reader = res.body.getReader();
            for (;;) {
                const { done, value } = await reader.read();
                if (done) break;
                received += value.length;
                if (!out.write(value)) await new Promise((r) => out.once("drain", r));
                if (total) {
                    const pct = Math.floor((received / total) * 100);
                    if (pct !== lastPct) {
                        lastPct = pct;
                        process.stdout.write(`\r  ${label} ${String(pct).padStart(3)}%  (${(received / 1048576).toFixed(1)}/${(total / 1048576).toFixed(1)} MB)`);
                    }
                }
            }
            await new Promise((r, j) => {
                out.once("error", j);
                out.once("close", r);
                out.end();
            });
            process.stdout.write("\n");
            if (total && received !== total) throw new Error(`size mismatch: got ${received}, expected ${total}`);
            renameSync(tmp, dest);
            return received;
        } catch (err) {
            rmSync(`${dest}.part`, { force: true });
            if (a >= attempts) throw new Error(`${err.message} for ${url}`);
            await new Promise((r) => setTimeout(r, 500 * a * a));
        }
    }
}

async function main() {
    const opts = parseArgs(process.argv.slice(2));
    if (opts.help) return usage();
    if (opts.list) return list();

    const wanted = opts.all
        ? GROUPS
        : opts.names.length
          ? opts.names.map((n) => {
                const g = GROUPS.find((x) => x.name === n.toLowerCase());
                if (!g) throw new Error(`unknown group "${n}" (known: ${GROUPS.map((x) => x.name).join(", ")})`);
                return g;
            })
          : bySmall();

    let downloaded = 0;
    let skipped = 0;
    for (const group of wanted) {
        const dir = join(opts.dest, group.dir);
        mkdirSync(dir, { recursive: true });
        console.log(`\n${group.name} → ${dir}`);
        console.log(`  ${group.note}`);
        for (const f of group.files) {
            const dest = join(dir, f.file);
            if (!opts.force && existsSync(dest) && statSync(dest).size > 0) {
                console.log(`  ${f.file} — already present (${(statSync(dest).size / 1048576).toFixed(1)} MB)`);
                skipped++;
                continue;
            }
            await downloadTo(f.url, dest, f.file);
            downloaded++;
        }
    }
    console.log(`\nDone: ${downloaded} downloaded, ${skipped} already present → ${opts.dest}`);
    console.log("For the generated formats (MERL BRDFs, IES profiles) run: node tools/gen-assets.mjs");
}

main().catch((err) => {
    console.error(`\nerror: ${err.message}`);
    process.exit(1);
});
