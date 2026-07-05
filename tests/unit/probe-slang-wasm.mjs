// Probe: slang-wasm runtime compilation to WGSL with Falcor module imports via MEMFS.
import { readFileSync } from "node:fs";

const { default: factory } = await import("../../tools/slang-wasm/slang-wasm.js");
const slang = await factory();

console.log("compile targets:", JSON.stringify(slang.getCompileTargets?.()));

const globalSession = slang.createGlobalSession();
if (!globalSession) throw new Error("no global session");

// Find WGSL target id.
const targets = slang.getCompileTargets();
let wgslTarget = null;
for (const t of targets) if (String(t.name).toLowerCase().includes("wgsl")) wgslTarget = t.value;
console.log("WGSL target id:", wgslTarget);

const session = globalSession.createSession(wgslTarget);
if (!session) throw new Error("no session");

// Write a Falcor utility module into MEMFS and import it.
const hashUtils = readFileSync("Falcor/Source/Falcor/Utils/Math/HashUtils.slang", "utf8");
slang.FS.createPath("/", "Utils/Math", true, true);
slang.FS.writeFile("/Utils/Math/HashUtils.slang", hashUtils);

const source = `
import Utils.Math.HashUtils;
RWStructuredBuffer<uint> gOut;
[numthreads(64,1,1)]
void csMain(uint3 tid : SV_DispatchThreadID) { gOut[tid.x] = jenkinsHash(tid.x); }
`;

const module = session.loadModuleFromSource(source, "probe", "/probe.slang");
if (!module) {
    console.error("loadModuleFromSource failed:", slang.getLastError?.());
    process.exit(1);
}
const entryPoint = module.findAndCheckEntryPoint("csMain", 6 /* SLANG_STAGE_COMPUTE */);
if (!entryPoint) {
    console.error("entry point not found:", slang.getLastError?.());
    process.exit(1);
}
const composed = session.createCompositeComponentType([module, entryPoint]);
const linked = composed.link();
const wgsl = linked.getEntryPointCode(0, 0);
console.log("--- WGSL (first 400 chars) ---");
console.log(wgsl.slice(0, 400));

const layout = linked.getLayout(0);
const json = layout?.toJsonObject();
console.log("--- reflection keys:", Object.keys(json ?? {}));
console.log("--- params:", JSON.stringify(json?.parameters?.map((p) => ({ name: p.name, binding: p.binding }))));
