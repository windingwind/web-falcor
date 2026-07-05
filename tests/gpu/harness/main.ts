/**
 * Browser-side GPU test runner. Discovers *.gpu.test.ts files across the
 * workspace, runs them sequentially, reports results to the Node driver via
 * window.__results / console.
 */

import { Device, ProgramManager, initSlang } from "@web-falcor/falcor";
import { tests, SkipError } from "./registry.js";

/** Fetches the Falcor shader tree and wires the program system (M2+ tests). */
async function initProgramSystem(device: Device): Promise<void> {
    const list = (await (await fetch("/packages/falcor/shaders/generated/shader-file-list.json")).json()) as {
        falcorFiles: string[];
        renderPassFiles: string[];
        localFiles: string[];
    };
    const sources = new Map<string, string>();
    const missing: string[] = [];
    const fetchInto = async (urlBase: string, files: string[]) => {
        await Promise.all(
            files.map(async (f) => {
                const res = await fetch(`${urlBase}/${f}`);
                if (res.ok) sources.set(f, await res.text());
                else missing.push(`${urlBase}/${f} (${res.status})`);
            }),
        );
    };
    await Promise.all([
        fetchInto("/Falcor/Source/Falcor", list.falcorFiles),
        fetchInto("/Falcor/Source", list.renderPassFiles),
        fetchInto("/packages/falcor/shaders", list.localFiles),
    ]);
    if (missing.length > 0) {
        console.error(`shader registry: ${missing.length} files failed to fetch; first: ${missing.slice(0, 3).join(", ")}`);
    }
    await initSlang("/tools/slang-wasm/slang-wasm.js");
    device.setProgramManager(new ProgramManager(device, (p) => sources.get(p), [...sources.keys()]));
}

interface TestResult {
    name: string;
    status: "pass" | "fail" | "skip";
    error?: string;
    ms: number;
}

declare global {
    interface Window {
        __done: boolean;
        __results: TestResult[];
        __fatal?: string;
    }
}

const logEl = document.getElementById("log")!;
function log(line: string) {
    logEl.textContent += line + "\n";
    console.log(line);
}

async function run() {
    // Side-effect imports register tests into the registry.
    const modules = import.meta.glob(["/packages/*/src/**/*.gpu.test.ts", "/tests/gpu/suites/**/*.gpu.test.ts"]);
    for (const load of Object.values(modules)) await load();

    const device = await Device.create();
    const info = device.adapter.info;
    log(`# adapter: ${info?.vendor ?? "?"} ${info?.architecture ?? "?"}`);
    const t0 = performance.now();
    await initProgramSystem(device);
    log(`# program system ready in ${(performance.now() - t0).toFixed(0)}ms`);
    log(`# tests: ${tests.length}`);

    const results: TestResult[] = [];
    for (const test of tests) {
        const start = performance.now();
        try {
            await test.fn({ device });
            // Surface asynchronous validation errors per-test.
            await device.gpuDevice.queue.onSubmittedWorkDone();
            results.push({ name: test.name, status: "pass", ms: performance.now() - start });
            log(`PASS ${test.name}`);
        } catch (err) {
            if (err instanceof SkipError) {
                results.push({ name: test.name, status: "skip", error: err.message, ms: performance.now() - start });
                log(`SKIP ${test.name} (${err.message})`);
            } else {
                const msg = err instanceof Error ? (err.stack ?? err.message) : String(err);
                results.push({ name: test.name, status: "fail", error: msg, ms: performance.now() - start });
                log(`FAIL ${test.name}: ${msg}`);
            }
        }
    }
    window.__results = results;
    window.__done = true;
}

run().catch((err) => {
    window.__fatal = err instanceof Error ? (err.stack ?? err.message) : String(err);
    window.__done = true;
    log(`FATAL ${window.__fatal}`);
});
