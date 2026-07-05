/**
 * Browser-side GPU test runner. Discovers *.gpu.test.ts files across the
 * workspace, runs them sequentially, reports results to the Node driver via
 * window.__results / console.
 */

import { Device } from "@web-falcor/falcor";
import { tests, SkipError } from "./registry.js";

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
