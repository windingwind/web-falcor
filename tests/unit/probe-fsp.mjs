import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
import { initSlang, SlangCompiler, ShaderType } from "../../packages/falcor/src/Core/Program/SlangCompiler.ts";
// Run via tsx? No — use vite-node? Simplest: this is TS import from mjs — won't work in plain node.
