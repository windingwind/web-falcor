#!/usr/bin/env node
/**
 * Mirrors Falcor/tools/make_new_render_pass.py: copies the RenderPassTemplate pass to
 * packages/render-passes/src/<Name>/<Name>.ts with every "RenderPassTemplate" renamed, and registers
 * it in the package index (native adds the directory to RenderPasses/CMakeLists.txt).
 *
 * Usage: node scripts/make-new-render-pass.mjs <Name>
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const kTemplate = "RenderPassTemplate";
const srcDir = join(dirname(fileURLToPath(import.meta.url)), "../packages/render-passes/src");

const name = process.argv[2];
if (!name || !/^[A-Za-z_]\w*$/.test(name)) {
    console.error("usage: node scripts/make-new-render-pass.mjs <Name>   (a valid identifier)");
    process.exit(1);
}
console.log(`Creating render pass library "${name}":`);
const dstDir = join(srcDir, name);
if (existsSync(dstDir)) {
    console.error(`"${name}" already exists!`);
    process.exit(1);
}
mkdirSync(dstDir);
const dstFile = join(dstDir, `${name}.ts`);
console.log(`Writing ${dstFile}.`);
writeFileSync(dstFile, readFileSync(join(srcDir, `${kTemplate}.ts`), "utf8").replaceAll(kTemplate, name));

// Register it: the index's side-effect imports are what add passes to the factory.
const indexFile = join(srcDir, "index.ts");
const index = readFileSync(indexFile, "utf8");
writeFileSync(indexFile, `${index.trimEnd()}\nexport * from "./${name}/${name}.js";\n`);
console.log(`Registered in ${indexFile}.`);
