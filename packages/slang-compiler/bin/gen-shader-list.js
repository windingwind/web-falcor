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

const isShader = (name) => name.endsWith(".slang") || name.endsWith(".slangh");

const falcorFiles = walk(falcorRoot, isShader).map((p) => relative(falcorRoot, p).replaceAll("\\", "/"));
const localFiles = walk(localRoot, isShader).map((p) => relative(localRoot, p).replaceAll("\\", "/"));

const outDir = join(localRoot, "generated");
mkdirSync(outDir, { recursive: true });
writeFileSync(join(outDir, "shader-file-list.json"), JSON.stringify({ falcorFiles, localFiles }, null, 1));
console.log(`shader-file-list.json: ${falcorFiles.length} Falcor + ${localFiles.length} local shader files`);
