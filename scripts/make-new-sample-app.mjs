#!/usr/bin/env node
/**
 * Mirrors Falcor/tools/make_new_sample_app.py: copies SampleAppTemplate to
 * packages/samples/src/<Name>.ts with every "SampleAppTemplate" renamed, and adds it to the sample
 * launcher's kSamples (run it with `?sample=<Name>`; native adds it to Samples/CMakeLists.txt).
 *
 * Usage: node scripts/make-new-sample-app.mjs <Name>
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const kTemplate = "SampleAppTemplate";
const srcDir = join(dirname(fileURLToPath(import.meta.url)), "../packages/samples/src");

const name = process.argv[2];
if (!name || !/^[A-Za-z_]\w*$/.test(name)) {
    console.error("usage: node scripts/make-new-sample-app.mjs <Name>   (a valid identifier)");
    process.exit(1);
}
console.log(`Creating sample application "${name}":`);
const dstFile = join(srcDir, `${name}.ts`);
if (existsSync(dstFile)) {
    console.error(`"${name}" already exists!`);
    process.exit(1);
}
console.log(`Writing ${dstFile}.`);
writeFileSync(dstFile, readFileSync(join(srcDir, `${kTemplate}.ts`), "utf8").replaceAll(kTemplate, name));

// Register it with the launcher: an import next to the template's and a kSamples entry.
const mainFile = join(srcDir, "main.ts");
let main = readFileSync(mainFile, "utf8");
const importLine = `import { ${kTemplate} } from "./${kTemplate}.js";`;
const samples = /export const kSamples: Record<string, SampleClass> = \{ ([^}]*) \};/;
if (!main.includes(importLine) || !samples.test(main)) {
    console.error(`Couldn't find the template's import or kSamples in ${mainFile}; register ${name} by hand.`);
    process.exit(1);
}
main = main.replace(importLine, `${importLine}\nimport { ${name} } from "./${name}.js";`);
main = main.replace(samples, (_m, list) => `export const kSamples: Record<string, SampleClass> = { ${list}, ${name} };`);
writeFileSync(mainFile, main);
console.log(`Registered in ${mainFile}.`);
