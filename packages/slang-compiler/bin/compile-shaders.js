#!/usr/bin/env node
/**
 * Slang -> WGSL shader compilation driver.
 *
 * Compiles entries listed in a shader manifest (JSON) with slangc, emitting for each:
 *   <out>/<name>.wgsl            WGSL source
 *   <out>/<name>.reflection.json Slang reflection (parameter blocks, offsets, bindings)
 *
 * Manifest entry: { "source": "...", "entry": "main", "stage": "compute", "defines": {...}, "name": "..." }
 * Include paths default to Falcor/Source/Falcor (upstream shader library) plus the manifest's directory.
 *
 * This is the build-time path. The runtime path (in-browser compilation of user
 * shaders with slang-wasm) shares manifest semantics; see docs §Shader system.
 */

import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const slangc = join(repoRoot, "tools/slang/bin/slangc");
const falcorShaderRoot = join(repoRoot, "Falcor/Source/Falcor");

function compileEntry(entry, outDir, extraIncludes = []) {
    const name = entry.name ?? entry.entry;
    const outWgsl = join(outDir, `${name}.wgsl`);
    const outRefl = join(outDir, `${name}.reflection.json`);
    const args = [
        entry.source,
        "-target", "wgsl",
        "-entry", entry.entry,
        "-stage", entry.stage,
        "-I", falcorShaderRoot,
        ...extraIncludes.flatMap((p) => ["-I", p]),
        ...Object.entries(entry.defines ?? {}).flatMap(([k, v]) => ["-D", v === "" ? k : `${k}=${v}`]),
        "-reflection-json", outRefl,
        "-o", outWgsl,
    ];
    execFileSync(slangc, args, { stdio: "inherit" });
    return { name, wgsl: outWgsl, reflection: outRefl };
}

function main() {
    const manifestPath = process.argv[2] ?? join(repoRoot, "packages/falcor/shaders/manifest.json");
    if (!existsSync(manifestPath)) {
        console.error(`Shader manifest not found: ${manifestPath}`);
        process.exit(1);
    }
    const manifestDir = dirname(resolve(manifestPath));
    const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
    const outDir = resolve(manifestDir, manifest.outDir ?? "generated");
    mkdirSync(outDir, { recursive: true });

    let failed = 0;
    for (const entry of manifest.entries) {
        try {
            const src = resolve(manifestDir, entry.source);
            const result = compileEntry({ ...entry, source: src }, outDir, [manifestDir]);
            console.log(`[ok] ${result.name}`);
        } catch (err) {
            console.error(`[fail] ${entry.name ?? entry.entry}: ${err.message}`);
            failed++;
        }
    }
    process.exit(failed > 0 ? 1 : 0);
}

main();
