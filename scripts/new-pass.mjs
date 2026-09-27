#!/usr/bin/env node
/**
 * Scaffolds a render pass: a compute pass (TS + Slang) and a Mogwai graph script that runs it.
 *
 *   npm run new:pass -- <Name>            in-tree, under packages/render-passes/src/<Name>/
 *   npm run new:pass -- <Name> --plugin   out-of-tree plugin, under plugins/<Name>/ (loaded with ?plugin=)
 * The templates live in packages/mogwai/lib/scaffold.mjs, shared with the npm CLI's `web-falcor new`.
 */

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync, appendFileSync } from "node:fs";
import { join, relative, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { refreshShaderList } from "../packages/slang-compiler/bin/repo-shader-list.js";
import { isPassName, passFiles } from "../packages/mogwai/lib/scaffold.mjs";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const passesSrc = join(repoRoot, "packages/render-passes/src");

function usage(msg) {
    if (msg) console.error(`error: ${msg}\n`);
    console.error("usage: npm run new:pass -- <Name> [--plugin]\n  <Name>: PascalCase pass type, e.g. MyBlur");
    process.exit(msg ? 1 : 0);
}

const args = process.argv.slice(2);
if (args.includes("-h") || args.includes("--help")) usage();
const plugin = args.includes("--plugin");
const unknown = args.filter((a) => a.startsWith("-") && a !== "--plugin");
if (unknown.length > 0) usage(`unknown option ${unknown[0]}`);
const names = args.filter((a) => !a.startsWith("-"));
if (names.length !== 1) usage("give exactly one pass name");
const name = names[0];
if (!isPassName(name)) usage(`'${name}' is not a PascalCase identifier`);

/** Pass types already registered in-tree (registerRenderPass("X", ...)). */
function registeredPassTypes() {
    const types = new Set();
    const walk = (dir) => {
        for (const e of readdirSync(dir, { withFileTypes: true })) {
            const p = join(dir, e.name);
            if (e.isDirectory()) walk(p);
            else if (e.name.endsWith(".ts")) for (const m of readFileSync(p, "utf8").matchAll(/registerRenderPass\(\s*"([^"]+)"/g)) types.add(m[1]);
        }
    };
    walk(passesSrc);
    return types;
}
if (registeredPassTypes().has(name)) usage(`a pass named '${name}' is already registered in packages/render-passes`);

const outDir = plugin ? join(repoRoot, "plugins", name) : join(passesSrc, name);
if (existsSync(outDir)) usage(`${relative(repoRoot, outDir)} already exists`);

mkdirSync(outDir, { recursive: true });
const written = passFiles(name, plugin ? "repo-plugin" : "in-tree").map(([file, text]) => {
    writeFileSync(join(outDir, file), text);
    return relative(repoRoot, join(outDir, file));
});

if (!plugin) {
    appendFileSync(join(passesSrc, "index.ts"), `export * from "./${name}/${name}.js";\n`);
    written.push("packages/render-passes/src/index.ts (export added)");
    refreshShaderList();
    written.push("packages/falcor/shaders/generated/shader-file-list.json (shader registered)");
}

const dir = "/" + relative(repoRoot, outDir).replaceAll("\\", "/");
const url = `http://localhost:5173/?scene=test_scenes/cornell_box.pyscene&script=${dir}/${name}.graph.js${plugin ? `&plugin=${dir}/${name}.ts` : ""}`;
console.log(`Created ${plugin ? "plugin" : "in-tree"} pass ${name}:\n${written.map((f) => `  ${f}`).join("\n")}`);
console.log(`\nNext:\n  npm run dev\n  open ${url}`);
if (plugin) console.log(`\nplugins/ is gitignored: publish ${name} from its own repo (cd ${relative(repoRoot, outDir)} && git init).`);
