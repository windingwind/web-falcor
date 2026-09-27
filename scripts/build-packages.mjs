#!/usr/bin/env node
/**
 * Stages the npm packages under out/npm/ (the workspace itself keeps resolving the TypeScript sources):
 *   @web-falcor/falcor, @web-falcor/render-passes  compiled JS + type declarations (+ falcor's wasm modules)
 *   @web-falcor/mogwai                             the prebuilt viewer with the runtime assets, and the `web-falcor` CLI
 * Publish with `npm publish out/npm/<name> --access public`, falcor first. Needs `npm run setup:web` and test_scenes.
 */

import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = join(repoRoot, "out/npm");
const readJson = (p) => JSON.parse(readFileSync(join(repoRoot, p), "utf8"));
const node = (args, cwd = repoRoot) => execFileSync(process.execPath, args, { cwd, stdio: "inherit" });

const root = readJson("package.json");
const version = root.version;
for (const pkg of ["falcor", "render-passes", "mogwai"]) {
    const v = readJson(`packages/${pkg}/package.json`).version;
    if (v !== version) throw new Error(`packages/${pkg} is at ${v}, the repo at ${version}: bump them together`);
}

const common = {
    version,
    type: "module",
    license: "MIT",
    author: "windingwind",
    homepage: "https://github.com/windingwind/web-falcor#readme",
    repository: { type: "git", url: "git+https://github.com/windingwind/web-falcor.git" },
    bugs: "https://github.com/windingwind/web-falcor/issues",
    keywords: ["webgpu", "falcor", "rendering", "path-tracing", "slang", "render-graph"],
    publishConfig: { access: "public" },
};

function stage(name, pkgJson, readme) {
    const dir = join(outRoot, name);
    writeFileSync(join(dir, "package.json"), JSON.stringify({ name: `@web-falcor/${name}`, ...pkgJson, ...common }, null, 2) + "\n");
    for (const f of ["LICENSE", "THIRD-PARTY-NOTICES.md", "licenses"]) cpSync(join(repoRoot, f), join(dir, f), { recursive: true });
    writeFileSync(join(dir, "README.md"), readme);
    return dir;
}

/** Compiles a workspace package's src/ to <out>/dist (JS + .d.ts, no maps: the sources are not shipped). */
function compile(pkg, dir) {
    node([join(repoRoot, "node_modules/typescript/bin/tsc"), "-p", `packages/${pkg}/tsconfig.json`, "--noEmit", "false", "--outDir", join(dir, "dist"), "--sourceMap", "false", "--declarationMap", "false"]);
    // Plain-JS modules beside the TS sources.
    const walk = (rel) => {
        for (const name of readdirSync(join(repoRoot, `packages/${pkg}/src`, rel))) {
            const p = join(rel, name);
            const abs = join(repoRoot, `packages/${pkg}/src`, p);
            if (statSync(abs).isDirectory()) walk(p);
            else if (/\.(js|d\.ts)$/.test(name) && !name.endsWith(".ts.d.ts")) {
                mkdirSync(dirname(join(dir, "dist", p)), { recursive: true });
                cpSync(abs, join(dir, "dist", p));
            }
        }
    };
    walk("");
}

rmSync(outRoot, { recursive: true, force: true });
mkdirSync(outRoot, { recursive: true });

// @web-falcor/falcor
{
    const dir = join(outRoot, "falcor");
    mkdirSync(dir, { recursive: true });
    compile("falcor", dir);
    // The worker's URL names its source file; the compiled worker is .js.
    const pool = join(dir, "dist/Utils/Threading/WorkerPool.js");
    const text = readFileSync(pool, "utf8");
    if (!text.includes('"./TaskWorker.ts"')) throw new Error("WorkerPool.js: TaskWorker URL not found");
    writeFileSync(pool, text.replace('"./TaskWorker.ts"', '"./TaskWorker.js"'));
    cpSync(join(repoRoot, "packages/falcor/wasm"), join(dir, "wasm"), { recursive: true });
    const src = readJson("packages/falcor/package.json");
    stage(
        "falcor",
        {
            description: "web-falcor core: NVIDIA Falcor's framework on WebGPU (render graphs, scenes, materials, Slang shaders)",
            main: "./dist/index.js",
            types: "./dist/index.d.ts",
            exports: { ".": { types: "./dist/index.d.ts", import: "./dist/index.js" } },
            files: ["dist", "wasm", "licenses", "THIRD-PARTY-NOTICES.md"],
            sideEffects: true,
            dependencies: { "parse-exr": src.dependencies["parse-exr"] },
        },
        libraryReadme("falcor", "The core library: device, render graph, scene, materials and the Slang shader system."),
    );
}

// @web-falcor/render-passes
{
    const dir = join(outRoot, "render-passes");
    mkdirSync(dir, { recursive: true });
    compile("render-passes", dir);
    stage(
        "render-passes",
        {
            description: "web-falcor render passes: Falcor's PathTracer, GBuffer, AccumulatePass, ToneMapper, NRD, RTXDI and more on WebGPU",
            main: "./dist/index.js",
            types: "./dist/index.d.ts",
            exports: { ".": { types: "./dist/index.d.ts", import: "./dist/index.js" } },
            files: ["dist", "licenses", "THIRD-PARTY-NOTICES.md"],
            sideEffects: true,
            dependencies: { "@web-falcor/falcor": `^${version}` },
        },
        libraryReadme("render-passes", "Falcor's render passes. Importing the package registers them all with the render graph."),
    );
}

// @web-falcor/mogwai: the static viewer served from the package root (base "/").
{
    const dir = join(outRoot, "mogwai");
    node(["scripts/build-web.mjs", "--base", "/", "--out", join(dir, "dist"), "--no-media"]);
    for (const d of ["bin", "lib"]) cpSync(join(repoRoot, "packages/mogwai", d), join(dir, d), { recursive: true });
    stage(
        "mogwai",
        {
            description: "Mogwai, Falcor's interactive viewer, prebuilt for the browser (WebGPU); `npx @web-falcor/mogwai` serves it",
            bin: { "web-falcor": "./bin/web-falcor.mjs" },
            exports: { "./vite": { types: "./lib/vite.d.mts", import: "./lib/vite.mjs" } },
            files: ["dist", "bin", "lib", "licenses", "THIRD-PARTY-NOTICES.md"],
            peerDependencies: { vite: ">=5" },
            peerDependenciesMeta: { vite: { optional: true } },
        },
        readFileSync(join(repoRoot, "packages/mogwai/NPM-README.md"), "utf8"),
    );
}

console.log(`\nstaged in ${outRoot}:`);
for (const name of readdirSync(outRoot)) {
    const size = execFileSync("du", ["-sh", join(outRoot, name)], { encoding: "utf8" }).split("\t")[0];
    console.log(`  @web-falcor/${name}@${version} (${size})`);
}

function libraryReadme(name, blurb) {
    return `# @web-falcor/${name}

${blurb} Part of [web-falcor](https://github.com/windingwind/web-falcor), a WebGPU reimplementation of
[NVIDIA Falcor](https://github.com/NVIDIAGameWorks/Falcor).

See [Using the npm packages](https://github.com/windingwind/web-falcor/blob/main/docs/npm.md) for setup: the
library also needs its runtime assets (shaders, the Slang compiler, Pyodide), which ship in \`@web-falcor/mogwai\`.
`;
}
