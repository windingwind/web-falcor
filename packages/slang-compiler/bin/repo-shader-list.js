#!/usr/bin/env node
/**
 * Refreshes the repo-owned entries of shader-file-list.json (web shaders and in-tree pass shaders)
 * in place; the upstream Falcor lists stay as they are, so this works without a Falcor clone.
 */

import { existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const localRoot = join(repoRoot, "packages/falcor/shaders");
const passesRoot = join(repoRoot, "packages/render-passes/src");
const passesUrl = "/packages/render-passes/src/";
export const shaderListPath = join(localRoot, "generated/shader-file-list.json");

const isShader = (name) => name.endsWith(".slang") || name.endsWith(".slangh") || name.endsWith(".hlsli");

function walk(dir) {
    if (!existsSync(dir)) return [];
    const out = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const path = join(dir, entry.name);
        if (entry.isDirectory()) {
            if (entry.name !== "generated") out.push(...walk(path));
        } else if (isShader(entry.name)) {
            out.push(path);
        }
    }
    return out;
}

const rel = (root, p) => relative(root, p).replaceAll("\\", "/");

/** Web-owned shaders under packages/falcor/shaders, keyed relative to it. */
export function localShaderFiles() {
    return walk(localRoot).map((p) => rel(localRoot, p)).sort();
}

/** Shaders beside the in-tree passes, keyed like native's deployed RenderPasses/<Pass>/... layout. */
export function webPassShaderEntries() {
    return walk(passesRoot)
        .map((p) => rel(passesRoot, p))
        .sort()
        .map((f) => ({ path: `RenderPasses/${f}`, url: passesUrl + f }));
}

/** Rewrites localFiles and the in-tree pass entries of externalFiles; returns the new counts. */
export function refreshShaderList() {
    const list = JSON.parse(readFileSync(shaderListPath, "utf8"));
    const webPasses = webPassShaderEntries();
    const upstream = new Set(list.renderPassFiles);
    const clash = webPasses.find((e) => upstream.has(e.path));
    if (clash) throw new Error(`${clash.url} shadows the upstream shader ${clash.path}; rename the pass or add a shader override instead`);
    list.localFiles = localShaderFiles();
    list.externalFiles = [...(list.externalFiles ?? []).filter((e) => !e.url.startsWith(passesUrl)), ...webPasses];
    writeFileSync(shaderListPath, JSON.stringify(list, null, 2) + "\n");
    return { localFiles: list.localFiles.length, passShaders: webPasses.length };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const { localFiles, passShaders } = refreshShaderList();
    console.log(`shader-file-list.json: ${localFiles} local + ${passShaders} in-tree pass shader files (upstream lists unchanged)`);
}
