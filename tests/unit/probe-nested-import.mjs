// Reproduce nested-import resolution failure and test fixes.
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
console.log("cwd:", slang.FS.cwd ? slang.FS.cwd() : "n/a");

const target = slang.getCompileTargets().find((t) => t.name === "WGSL").value;
const session = slang.createGlobalSession().createSession(target);

for (const f of files) {
    const dir = f.split("/").slice(0, -1).join("/");
    if (dir) slang.FS.createPath("/", dir, true, true);
    slang.FS.writeFile(`/${f}`, readFileSync(join(root, f), "utf8"));
}

const entry = `
import Utils.Sampling.TinyUniformSampleGenerator;
RWStructuredBuffer<uint> gOut;
[numthreads(32,1,1)]
void main(uint3 tid : SV_DispatchThreadID) {
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(tid.xy), 1);
    gOut[tid.x] = sampleNext1D(sg) > 0.5 ? 1 : 0;
}`;

const mod = session.loadModuleFromSource(entry, "entry", "/entry.slang");
if (!mod) {
    console.log("FAIL:", JSON.stringify(slang.getLastError()));
    // Try fix: chdir
    if (slang.FS.chdir) {
        console.log("trying FS.chdir('/')...");
        slang.FS.chdir("/");
        const session2 = slang.createGlobalSession().createSession(target);
        const mod2 = session2.loadModuleFromSource(entry, "entry", "/entry.slang");
        console.log("after chdir:", mod2 ? "OK" : JSON.stringify(slang.getLastError()));
    }
} else {
    console.log("OK without fix");
}
