#!/usr/bin/env node
/**
 * Scene downloader — provisions Falcor's example scenes so you can try
 * web-falcor out of the box WITHOUT cloning or building upstream Falcor.
 *
 * The web app and the tests serve media from `Falcor/media/<Scene>/` (the dev
 * server exposes the repo root; `Falcor/` is .gitignored but present on disk).
 * `setup:web` deliberately does NOT fetch media — this script fills that gap.
 *
 * Two kinds of scene, one command:
 *
 *   • Bundled scenes (Arcade, test_scenes, …) come from Falcor's official media
 *     bundle, published by NVIDIA's packman CDN as a single ~120 MB `.7z` (the
 *     same artifact Falcor's `setup.sh` pulls). We download it once, cache it,
 *     and extract only the folders you ask for.
 *
 *   • ORCA showcase scenes (Bistro, Emerald Square, Sun Temple, Zero Day) are
 *     large, separately-hosted downloads. We fetch each one from NVIDIA's ORCA
 *     download endpoint (a stable `developer.nvidia.com/downloads/<slug>` URL
 *     that redirects to a signed `.zip`) and extract it into place.
 *
 * Usage:
 *   node scripts/download-scenes.mjs                 # all *bundled* scenes (out-of-box)
 *   node scripts/download-scenes.mjs Arcade Bistro   # only the named scene(s)
 *   node scripts/download-scenes.mjs --all           # everything, incl. the big ORCA scenes
 *   node scripts/download-scenes.mjs --list          # show the catalog and exit
 *   node scripts/download-scenes.mjs --force Arcade  # re-extract even if present
 *   node scripts/download-scenes.mjs --dest some/dir # extract elsewhere
 *   node scripts/download-scenes.mjs --clean         # delete cached archives after
 *
 * (via npm: `npm run download:scenes -- Bistro`)
 */

import { createWriteStream } from "node:fs";
import { mkdirSync, existsSync, statSync, rmSync, readdirSync, renameSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { execFileSync } from "node:child_process";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

// Pinned upstream media bundle (keep in sync with Falcor's dependencies.xml
// `falcor_media` version / README / DESIGN.md). The CDN + naming come from
// packman's config.packman.xml: `d4i3qtqj3r0z5.cloudfront.net/${name}@${version}`.
const FALCOR_MEDIA_VERSION = "7acdf8b0";
const MEDIA_ARCHIVE_URL = `https://d4i3qtqj3r0z5.cloudfront.net/falcor_media@${FALCOR_MEDIA_VERSION}.7z`;

// Scenes bundled in the packman media package (verified against the archive's
// top-level entries). Each maps to one top-level media folder; scenes inside a
// folder share assets (meshes/textures/envmaps) by relative path, so the folder
// is the smallest self-contained unit.
const BUNDLE = [
    { name: "test_scenes", sizeMB: 170, desc: "Cornell box (Mogwai's default), material/sphere/dielectric tests, curves, volumes, CesiumMan, …" },
    { name: "Arcade", sizeMB: 56, desc: "Textured arcade scene (fbx/obj/ply/pyscene) — the main lit demo and path-tracer oracle" },
    { name: "inv_rendering_scenes", sizeMB: 7, desc: "Inverse-rendering bunny/sphere scenes (init/ref pairs)" },
    { name: "test_images", sizeMB: 29, desc: "Reference images used by the image-comparison tests" },
];

// Large ORCA scenes, each a separate download from NVIDIA's ORCA archive.
// `slug` is the stable developer.nvidia.com/downloads/<slug> endpoint, which
// 302-redirects to a signed .zip; `folder` is where it lands under the dest.
const ORCA = [
    { name: "Bistro", folder: "Bistro_v5_2", slug: "bistro", sizeMB: 853, aliases: ["bistro_v5_2"], desc: "Amazon Lumberyard Bistro (interior + exterior) — the hero showcase scene" },
    { name: "EmeraldSquare", folder: "EmeraldSquare_v4_1", slug: "emerald-square", sizeMB: 589, aliases: ["emerald-square", "emerald_square", "emeraldsquare_v4_1"], desc: "NVIDIA Emerald Square — large outdoor city (day + dusk)" },
    { name: "SunTemple", folder: "SunTemple_v4", slug: "sun-temple", sizeMB: 336, aliases: ["sun-temple", "sun_temple", "suntemple_v4"], desc: "Unreal Engine Sun Temple" },
    { name: "ZeroDay", folder: "ZeroDay_v1", slug: "beeple", sizeMB: 1085, aliases: ["zero-day", "zero_day", "beeple", "zeroday_v1"], desc: "Beeple Zero-Day (Measure One + Measure Seven)" },
];

// Benedikt Bitterli's "Rendering Resources" — pbrt-v4 scenes, each a direct
// .zip at benedikt-bitterli.me/resources/pbrt-v4/<name>.zip (folder-wrapped),
// loadable via web-falcor's pbrt importer. This is a curated subset; ANY scene
// from that page also works by name (resolved on demand). name == folder == slug.
const PBRT_BASE = "https://benedikt-bitterli.me/resources/pbrt-v4";
const PBRT = [
    { name: "cornell-box", sizeMB: 3, desc: "Cornell box (verified) — diffuse walls + area light" },
    { name: "veach-mis", sizeMB: 2, desc: "Veach MIS (verified) — glossy conductor plates + area lights" },
    { name: "veach-ajar", sizeMB: 44, desc: "Veach ajar-door — hard indirect lighting" },
    { name: "living-room", sizeMB: 45, desc: "Modern living room" },
    { name: "staircase", sizeMB: 44, desc: "Wooden staircase" },
    { name: "kitchen", sizeMB: 41, desc: "Country kitchen" },
    { name: "bathroom", sizeMB: 44, desc: "Salle de bain (bathroom)" },
    { name: "bedroom", sizeMB: 51, desc: "Bedroom" },
    { name: "glass-of-water", sizeMB: 10, desc: "Glass of water (dielectric refraction)" },
    { name: "spaceship", sizeMB: 10, desc: "Spaceship" },
    { name: "coffee", sizeMB: 6, desc: "Coffee maker" },
    { name: "dragon", sizeMB: 14, desc: "Stanford dragon" },
];
const pbrtEntry = (name, sizeMB, desc) => ({ name, folder: name, slug: name, sizeMB, desc });

// name/alias/folder (lowercased) -> { scene, kind }
const lookup = new Map();
for (const s of BUNDLE) lookup.set(s.name.toLowerCase(), { scene: s, kind: "bundle" });
for (const s of ORCA) {
    for (const key of [s.name, s.folder, ...(s.aliases ?? [])]) lookup.set(key.toLowerCase(), { scene: s, kind: "orca" });
}
for (const s of PBRT) lookup.set(s.name.toLowerCase(), { scene: pbrtEntry(s.name, s.sizeMB, s.desc), kind: "pbrt" });
/** Destination folder name for a scene (bundle scenes use `name`). */
const folderOf = (s) => s.folder ?? s.name;

// ---------------------------------------------------------------------------

function parseArgs(argv) {
    const opts = { list: false, force: false, clean: false, help: false, all: false, dest: join(repoRoot, "Falcor/media"), names: [] };
    for (let i = 0; i < argv.length; i++) {
        const a = argv[i];
        if (a === "--list" || a === "-l") opts.list = true;
        else if (a === "--force" || a === "-f") opts.force = true;
        else if (a === "--clean") opts.clean = true;
        else if (a === "--all") opts.all = true;
        else if (a === "--help" || a === "-h") opts.help = true;
        else if (a === "--dest") opts.dest = resolve(argv[++i] ?? "");
        else if (a.startsWith("--dest=")) opts.dest = resolve(a.slice("--dest=".length));
        else if (a.startsWith("-")) throw new Error(`unknown flag: ${a} (try --help)`);
        else opts.names.push(a);
    }
    return opts;
}

function printHelp() {
    console.log(`Download Falcor example scenes into Falcor/media/ (no Falcor clone needed).

Usage:
  node scripts/download-scenes.mjs [scene...] [options]

  With no scene names, downloads every *bundled* scene (~120 MB, best out-of-box
  experience). Name scenes to fetch a subset (bundled, ORCA, or Bitterli pbrt-v4
  by name), or --all to also pull the large ORCA showcase scenes (~2.8 GB).

Options:
  -l, --list        List the available scenes and exit
      --all         Download everything, including the large ORCA scenes
  -f, --force       Re-extract scenes even if their folder already exists
      --dest <dir>  Extract into <dir> (default: Falcor/media)
      --clean       Delete cached archives when done
  -h, --help        Show this help

Examples:
  npm run download:scenes                    # all bundled scenes
  npm run download:scenes -- Arcade Bistro   # one bundled + one ORCA scene
  npm run download:scenes -- cornell-box     # a Bitterli pbrt-v4 scene
  npm run download:scenes -- --all           # everything
  npm run download:scenes -- --list`);
}

function printList() {
    console.log("Bundled scenes — one shared ~120 MB download (Falcor media bundle):\n");
    for (const s of BUNDLE) console.log(`  ${s.name.padEnd(22)} ${String(s.sizeMB).padStart(4)} MB   ${s.desc}`);
    console.log("\nORCA showcase scenes — large, downloaded individually from NVIDIA ORCA:\n");
    for (const s of ORCA) console.log(`  ${s.name.padEnd(22)} ${String(s.sizeMB).padStart(4)} MB   ${s.desc}`);
    console.log("\nBitterli pbrt-v4 scenes (loaded via the pbrt importer; any name from the");
    console.log("Rendering Resources page also works, not just those listed):\n");
    for (const s of PBRT) console.log(`  ${s.name.padEnd(22)} ${String(s.sizeMB).padStart(4)} MB   ${s.desc}`);
    console.log("\n(ORCA scenes are under NVIDIA ORCA licenses: https://developer.nvidia.com/orca;");
    console.log(" Bitterli scenes are CC0/CC-BY: https://benedikt-bitterli.me/resources)");
}

/** Locate a 7-Zip CLI (for the .7z media bundle), preferring PATH then packman. */
function find7z() {
    for (const bin of ["7z", "7za", "7zr"]) {
        try {
            execFileSync(bin, ["--help"], { stdio: "ignore" });
            return bin;
        } catch {
            /* not on PATH */
        }
    }
    const packmanChk = join(process.env.HOME ?? "", ".cache/packman/chk/7za");
    if (existsSync(packmanChk)) {
        for (const ver of readdirSync(packmanChk)) {
            const p = join(packmanChk, ver, "7za");
            if (existsSync(p)) return p;
        }
    }
    return null;
}

function haveUnzip() {
    try {
        execFileSync("unzip", ["-v"], { stdio: "ignore" });
        return true;
    } catch {
        return false;
    }
}

function toolHint(tool) {
    const pkg = tool === "unzip"
        ? { deb: "unzip", rpm: "unzip", brew: "unzip", win: "info-zip / built-in tar" }
        : { deb: "p7zip-full", rpm: "p7zip p7zip-plugins", brew: "p7zip", win: "7zip.7zip" };
    return [
        `\`${tool}\` is required to extract this scene but was not found. Install it:`,
        `  Debian/Ubuntu:  sudo apt-get install -y ${pkg.deb}`,
        `  Fedora/RHEL:    sudo dnf install -y ${pkg.rpm}`,
        `  macOS:          brew install ${pkg.brew}`,
        `  Windows:        winget install ${pkg.win}   (or https://www.7-zip.org)`,
    ].join("\n");
}

async function head(url) {
    try {
        const res = await fetch(url, { method: "HEAD" });
        if (!res.ok) return null;
        const len = Number(res.headers.get("content-length"));
        return Number.isFinite(len) && len > 0 ? len : null;
    } catch {
        return null;
    }
}

/** Resolve an ORCA download slug to its (signed, time-limited) .zip URL. */
async function orcaSignedUrl(slug) {
    const res = await fetch(`https://developer.nvidia.com/downloads/${slug}`, { redirect: "manual" });
    const loc = res.headers.get("location");
    if (!loc) throw new Error(`no download redirect for "${slug}" (HTTP ${res.status}) — the ORCA endpoint may have changed`);
    return loc;
}

/** Stream a URL to `dest` with a progress bar and retry-with-backoff. */
async function downloadTo(url, dest, expectedBytes, label, attempts = 4) {
    for (let a = 1; ; a++) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const total = Number(res.headers.get("content-length")) || expectedBytes || 0;
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
                        process.stdout.write(`\r  ${label} ${String(pct).padStart(3)}%  (${(received / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)} MB)`);
                    }
                }
            }
            // Wait for the fd to actually close (the 'close' event, not just
            // 'finish') before renaming/removing — otherwise a rename-then-unlink
            // of a still-open handle silly-renames to a stray .nfs* file on NFS.
            await new Promise((r, j) => {
                out.once("error", j);
                out.once("close", r);
                out.end();
            });
            process.stdout.write("\n");
            if (total && received !== total) throw new Error(`size mismatch: got ${received}, expected ${total}`);
            renameSync(tmp, dest);
            return;
        } catch (err) {
            rmSync(`${dest}.part`, { force: true });
            if (a >= attempts) throw new Error(`${err.message} for ${url}`);
            await new Promise((r) => setTimeout(r, 500 * a * a));
        }
    }
}

function dirHasFiles(p) {
    try {
        return statSync(p).isDirectory() && readdirSync(p).length > 0;
    } catch {
        return false;
    }
}

/**
 * Extract a downloaded scene .zip so the payload lands at `<dest>/<folder>`,
 * regardless of whether the zip wraps a single top-level folder (Bistro/…) or
 * stores files at the root (Zero Day).
 */
export function extractZip(zipPath, dest, folder) {
    const staging = join(dest, `.dl-${folder}`);
    rmSync(staging, { recursive: true, force: true });
    mkdirSync(staging, { recursive: true });
    execFileSync("unzip", ["-q", "-o", zipPath, "-d", staging], { stdio: "ignore" });

    const target = join(dest, folder);
    rmSync(target, { recursive: true, force: true });
    const children = readdirSync(staging);
    if (children.length === 1 && statSync(join(staging, children[0])).isDirectory()) {
        // zip wrapped everything in one folder — promote it to <dest>/<folder>.
        renameSync(join(staging, children[0]), target);
        rmSync(staging, { recursive: true, force: true });
    } else {
        // flat zip — the staging dir itself is the scene.
        renameSync(staging, target);
    }
    if (!dirHasFiles(target)) throw new Error(`extraction produced no files for ${folder}`);
}

// ---------------------------------------------------------------------------

async function main() {
    const opts = parseArgs(process.argv.slice(2));
    if (opts.help) return printHelp();
    if (opts.list) return printList();

    // Resolve the requested scenes.
    //   (no names, no --all) -> all bundled scenes (the out-of-box set)
    //   --all                -> everything: bundled + ORCA + curated pbrt
    //   names                -> exactly those (bundled, ORCA, or any Bitterli pbrt scene)
    let requested;
    if (opts.all) {
        requested = [
            ...BUNDLE.map((scene) => ({ scene, kind: "bundle" })),
            ...ORCA.map((scene) => ({ scene, kind: "orca" })),
            ...PBRT.map((s) => ({ scene: pbrtEntry(s.name, s.sizeMB, s.desc), kind: "pbrt" })),
        ];
    } else if (opts.names.length === 0) {
        requested = BUNDLE.map((scene) => ({ scene, kind: "bundle" }));
    } else {
        requested = [];
        const unknown = [];
        for (const raw of opts.names) {
            const hit = lookup.get(raw.toLowerCase());
            if (hit) requested.push(hit);
            else unknown.push(raw);
        }
        // An unknown name may still be a Bitterli pbrt-v4 scene (the catalog only
        // lists a curated subset) — probe the resource server before giving up.
        const stillUnknown = [];
        for (const u of unknown) {
            const bytes = await head(`${PBRT_BASE}/${u}.zip`);
            if (bytes) requested.push({ scene: pbrtEntry(u, Math.max(1, Math.round(bytes / 1048576)), "Bitterli pbrt-v4 scene"), kind: "pbrt" });
            else stillUnknown.push(u);
        }
        if (stillUnknown.length > 0) {
            for (const u of stillUnknown) console.error(`unknown scene: "${u}"`);
            console.error(`\nRun with --list to see the available scenes.`);
            process.exitCode = 1;
            return;
        }
    }

    // Skip scenes that are already present (unless --force).
    const todo = requested.filter(({ scene }) => {
        const present = dirHasFiles(join(opts.dest, folderOf(scene)));
        if (present && !opts.force) {
            console.log(`✓ ${scene.name} already present — skipping (use --force to re-extract)`);
            return false;
        }
        return true;
    });
    if (todo.length === 0) {
        console.log("\nNothing to do. All requested scenes are already present.");
        return;
    }

    // Check the tools we'll actually need up front (7z for the bundle .7z,
    // unzip for the per-scene ORCA/pbrt .zip archives).
    const needBundle = todo.some((t) => t.kind === "bundle");
    const needZip = todo.some((t) => t.kind === "orca" || t.kind === "pbrt");
    const sevenZip = needBundle ? find7z() : null;
    if (needBundle && !sevenZip) {
        console.error(toolHint("7z"));
        process.exitCode = 1;
        return;
    }
    if (needZip && !haveUnzip()) {
        console.error(toolHint("unzip"));
        process.exitCode = 1;
        return;
    }

    const cacheDir = join(repoRoot, "tools/cache");
    mkdirSync(cacheDir, { recursive: true });
    mkdirSync(opts.dest, { recursive: true });
    const usedArchives = [];

    const orcaTotal = todo.filter((t) => t.kind === "orca").reduce((n, t) => n + t.scene.sizeMB, 0);
    if (orcaTotal > 0) console.log(`Note: the ORCA scene(s) selected total ~${(orcaTotal / 1024).toFixed(1)} GB to download.`);

    // --- Bundled scenes: one shared .7z, selective extract. ---
    if (needBundle) {
        const archivePath = join(cacheDir, `falcor_media@${FALCOR_MEDIA_VERSION}.7z`);
        usedArchives.push(archivePath);
        const expected = await head(MEDIA_ARCHIVE_URL);
        const cached = existsSync(archivePath) && (!expected || statSync(archivePath).size === expected);
        if (cached) console.log(`Using cached media bundle (${(statSync(archivePath).size / 1048576).toFixed(0)} MB)`);
        else {
            console.log(`Downloading Falcor media bundle @ ${FALCOR_MEDIA_VERSION} (~120 MB, one time)…`);
            await downloadTo(MEDIA_ARCHIVE_URL, archivePath, expected, "media");
        }
        for (const { scene } of todo.filter((t) => t.kind === "bundle")) {
            process.stdout.write(`  extract ${scene.name} (~${scene.sizeMB} MB)… `);
            rmSync(join(opts.dest, scene.name), { recursive: true, force: true });
            execFileSync(sevenZip, ["x", "-y", `-o${opts.dest}`, archivePath, `${scene.name}/*`, "LICENSE.md"], { stdio: "ignore" });
            if (!dirHasFiles(join(opts.dest, scene.name))) throw new Error(`extraction produced no files for ${scene.name}`);
            console.log("done");
        }
    }

    // --- ORCA + Bitterli pbrt scenes: one .zip each. ---
    for (const { scene, kind } of todo.filter((t) => t.kind === "orca" || t.kind === "pbrt")) {
        const label = kind === "orca" ? "ORCA" : "pbrt";
        console.log(`\n${scene.name} (${scene.folder}) — resolving ${label} download…`);
        const url = kind === "orca" ? await orcaSignedUrl(scene.slug) : `${PBRT_BASE}/${scene.slug}.zip`;
        const zipPath = join(cacheDir, `${scene.folder}.zip`);
        usedArchives.push(zipPath);
        const expected = await head(url);
        const cached = existsSync(zipPath) && (!expected || statSync(zipPath).size === expected);
        if (cached) console.log(`  using cached ${scene.folder}.zip (${(statSync(zipPath).size / 1048576).toFixed(0)} MB)`);
        else await downloadTo(url, zipPath, expected, scene.name);
        process.stdout.write(`  extracting ${scene.folder}… `);
        extractZip(zipPath, opts.dest, scene.folder);
        console.log("done");
    }

    if (opts.clean) {
        for (const a of usedArchives) rmSync(a, { force: true });
        console.log(`\nRemoved ${usedArchives.length} cached archive(s).`);
    } else if (usedArchives.length > 0) {
        console.log(`\nCached archive(s) kept under ${cacheDir} (delete them or re-run with --clean to reclaim space).`);
    }

    console.log(`\nDone. Scenes are under ${opts.dest} and served by the dev server at /Falcor/media/<Scene>/.`);
    console.log(`Try: npm run dev   (Mogwai loads test_scenes/cornell_box.pyscene by default)`);
}

// Run only when invoked as a script (not when imported for testing).
if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
    main().catch((err) => {
        console.error(`\nError: ${err.message}`);
        process.exitCode = 1;
    });
}
