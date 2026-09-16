/**
 * Shader manifest vs what setup:web provisions: every registry file must exist
 * at its served path, and every SDK header the shaders include must be listed
 * in externalFiles with the upstream URL setup:web fetches it from.
 */

import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const manifest = JSON.parse(readFileSync(resolve(repoRoot, "packages/falcor/shaders/generated/shader-file-list.json"), "utf8")) as {
    falcorFiles: string[];
    renderPassFiles: string[];
    localFiles: string[];
    externalFiles: { path: string; url: string; upstream?: string }[];
};

// Registry key + served path per entry, mirroring initProgramSystem in mogwai/main.ts.
const entries = [
    ...manifest.falcorFiles.map((f) => ({ key: f, file: `Falcor/Source/Falcor/${f}` })),
    ...manifest.renderPassFiles.map((f) => ({ key: f, file: `Falcor/Source/${f}` })),
    ...manifest.localFiles.map((f) => ({ key: f, file: `packages/falcor/shaders/${f}` })),
    ...manifest.externalFiles.map((e) => ({ key: e.path, file: e.url.replace(/^\//, "") })),
];

// Needs the fetched Falcor shader tree (setup:web or a full clone); skip otherwise.
const hasShaders = existsSync(resolve(repoRoot, "Falcor/Source/Falcor/Scene/Scene.slang"));

describe.skipIf(!hasShaders)("shader manifest", () => {
    it("every registry file exists at its served path", () => {
        const missing = entries.filter((e) => !existsSync(resolve(repoRoot, e.file))).map((e) => e.file);
        expect(missing).toEqual([]);
    });

    it("every SDK header the shaders include is an externalFiles entry", () => {
        const keys = new Set(entries.map((e) => e.key));
        const external = new Set(manifest.externalFiles.map((e) => e.path));
        const unlisted = new Set<string>();
        for (const e of entries) {
            const file = resolve(repoRoot, e.file);
            if (!existsSync(file)) continue; // reported by the test above
            const dir = e.key.split("/").slice(0, -1).join("/");
            for (const m of readFileSync(file, "utf8").matchAll(/^\s*#\s*include\s+"([^"]+)"/gm)) {
                // Same resolution as SlangCompiler.rewriteIncludes: root-relative, then dir-relative.
                const target = m[1]!;
                const resolved = keys.has(target) ? target : keys.has(`${dir}/${target}`) ? `${dir}/${target}` : target;
                if (/^(nanovdb|rtxdi)\//.test(resolved) && !external.has(resolved)) unlisted.add(`${e.key}: ${target}`);
            }
        }
        expect([...unlisted]).toEqual([]);
    });

    it("externalFiles carry the upstream URL setup:web fetches from", () => {
        for (const e of manifest.externalFiles) expect(e.upstream, e.path).toMatch(/^https:\/\//);
    });
});
