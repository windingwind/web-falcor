#!/usr/bin/env node
/**
 * Generates the shader source file list consumed by the browser-side program
 * system (fetched into slang-wasm's MEMFS). Lists all .slang/.slangh files in
 * the upstream Falcor shader tree plus web-falcor's own shader dir.
 */

import { readdirSync, writeFileSync, mkdirSync } from "node:fs";
import { join, relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const falcorRoot = join(repoRoot, "Falcor/Source/Falcor");
// Falcor's deployed shader layout merges Source/Falcor/* with Source/RenderPasses -> RenderPasses/*.
const renderPassesRoot = join(repoRoot, "Falcor/Source/RenderPasses");
const localRoot = join(repoRoot, "packages/falcor/shaders");

function walk(dir, filter) {
    const out = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name === "generated") continue;
            out.push(...walk(path, filter));
        } else if (filter(entry.name)) {
            out.push(path);
        }
    }
    return out;
}

const isShader = (name) => name.endsWith(".slang") || name.endsWith(".slangh") || name.endsWith(".hlsli");

// SDK headers the upstream shaders include, served from Falcor's packman link paths (`url`).
// `upstream`: public byte-identical copies scripts/setup-web.mjs fetches for the no-clone setup.
const NANOVDB_RAW = "https://raw.githubusercontent.com/AcademySoftwareFoundation/openvdb/v9.1.0/nanovdb/nanovdb"; // = packman nanovdb 32.3.3 (MPL-2.0)
const RTXDI_RAW = "https://raw.githubusercontent.com/NVIDIAGameWorks/RTXDI/v1.3.0/rtxdi-sdk/include/rtxdi"; // = packman rtxdi 1.3.0-falcor
const externalFiles = [
    { path: "nanovdb/PNanoVDB.h", url: "/Falcor/external/packman/nanovdb/include/nanovdb/PNanoVDB.h", upstream: `${NANOVDB_RAW}/PNanoVDB.h` },
    ...["RTXDI.h", "ResamplingFunctions.hlsli", "Reservoir.hlsli", "RtxdiHelpers.hlsli", "RtxdiMath.hlsli", "RtxdiParameters.h", "RtxdiTypes.h"].map((f) => ({
        path: `rtxdi/${f}`,
        url: `/Falcor/external/packman/rtxdi/rtxdi-sdk/include/rtxdi/${f}`,
        upstream: `${RTXDI_RAW}/${f}`,
    })),
];

// Codepoint-sorted so the manifest is stable across filesystems (it is tracked like a lockfile).
const list = (root, prefix) => walk(root, isShader).map((p) => prefix + relative(root, p).replaceAll("\\", "/")).sort();
const falcorFiles = list(falcorRoot, "");
const renderPassFiles = list(renderPassesRoot, "RenderPasses/");
const localFiles = list(localRoot, "");

const outDir = join(localRoot, "generated");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "shader-file-list.json"), JSON.stringify({ falcorFiles, renderPassFiles, localFiles, externalFiles }, null, 2) + "\n");
console.log(`shader-file-list.json: ${falcorFiles.length} Falcor + ${renderPassFiles.length} render-pass + ${localFiles.length} local + ${externalFiles.length} external shader files`);
