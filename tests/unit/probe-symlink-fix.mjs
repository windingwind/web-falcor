import { readFileSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";
const { default: factory } = await import("../../tools/slang-wasm/slang-wasm.js");
const slang = await factory();
const root = "Falcor/Source/Falcor";
function walk(dir) {
    const out = [];
    for (const e of readdirSync(dir, { withFileTypes: true })) {
        const p = join(dir, e.name);
        if (e.isDirectory()) out.push(...walk(p));
        else if (e.name.endsWith(".slang") || e.name.endsWith(".slangh")) out.push(relative(root, p).replaceAll("\\", "/"));
    }
    return out;
}
const files = walk(root);
const target = slang.getCompileTargets().find((t) => t.name === "WGSL").value;
const session = slang.createGlobalSession().createSession(target);
const dirs = new Set();
for (const f of files) {
    const dir = f.split("/").slice(0, -1).join("/");
    if (dir) { slang.FS.createPath("/", dir, true, true); dirs.add(dir); }
    slang.FS.writeFile(`/${f}`, readFileSync(join(root, f), "utf8"));
}
// Symlink top-level shader roots into every directory so dir-relative import
// resolution finds root-relative module paths.
const roots = [...new Set(files.map((f) => f.split("/")[0]))];
console.log("top roots:", roots.join(","));
for (const d of dirs) {
    for (const r of roots) {
        const linkPath = `/${d}/${r}`;
        if (d.split("/")[0] === r && d.split("/").length === 1) continue; // don't self-link /Utils/Utils→ok actually needed
        try { slang.FS.symlink(`/${r}`, linkPath); } catch { /* exists as real dir */ }
    }
}
const entry = `
import Utils.Sampling.TinyUniformSampleGenerator;
RWStructuredBuffer<uint> gOut;
[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(tid.xy), 1);
    gOut[tid.x] = uint(sampleNext1D(sg) * 100.0);
}`;
const mod = session.loadModuleFromSource(entry, "entry", "/entry.slang");
if (!mod) { console.log("STILL FAIL:", JSON.stringify(slang.getLastError()).slice(0, 300)); process.exit(1); }
const ep = mod.findAndCheckEntryPoint("main", 6);
const linked = session.createCompositeComponentType([mod, ep]).link();
const wgsl = linked.getEntryPointCode(0, 0);
console.log("OK, WGSL bytes:", wgsl.length);
