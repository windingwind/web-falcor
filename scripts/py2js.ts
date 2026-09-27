/**
 * Converts Python scene/graph/Mogwai scripts to JS script modules (packages/falcor/src/Utils/Scripting/PyToJs.ts).
 *
 *   npm run py2js -- <file-or-dir>... [--out <dir>] [--root <dir>] [--check]
 *
 * The same driver is `npx @web-falcor/mogwai py2js` (packages/mogwai/lib/py2js.mjs). --check syntax-checks every
 * output with Node and exits non-zero on warnings.
 */

import { dirname } from "node:path";
import { createRequire } from "node:module";
import * as core from "../packages/falcor/src/Utils/Scripting/PyToJs.js";
import { convertFiles, parseArgs } from "../packages/mogwai/lib/py2js.mjs";

const args = parseArgs(process.argv.slice(2));
if (!args.inputs.length) {
    console.error("usage: npm run py2js -- <file-or-dir>... [--out <dir>] [--root <dir>] [--check]");
    process.exit(1);
}
const pyodideDir = dirname(createRequire(import.meta.url).resolve("pyodide/package.json"));
const warnings = await convertFiles({ ...args, core, pyodideDir });
if (args.check && warnings) process.exit(1);
