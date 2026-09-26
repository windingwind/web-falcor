#!/usr/bin/env node
/**
 * GPU test driver: serves the workspace with Vite, launches Chromium
 * (hardware WebGPU via Vulkan; requires an X display — run under `xvfb-run -a`),
 * executes the browser-side runner, reports results.
 *
 * Usage: xvfb-run -a node tests/gpu/harness/run.mjs [--swiftshader] [--filter <substring>[|<substring>...]]
 */

import { createServer } from "vite";
import { chromium } from "playwright";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync, realpathSync } from "node:fs";
import { dirListing } from "../../../scripts/vite-plugin-dir-listing.mjs";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const useSwiftShader = process.argv.includes("--swiftshader");
const filterIdx = process.argv.indexOf("--filter");
const filter = filterIdx >= 0 ? process.argv[filterIdx + 1] : undefined;

const vite = await createServer({
    root: repoRoot,
    // No HMR or file watching: an edit during a run must not reload or patch the test page.
    server: { port: 0, host: "127.0.0.1", hmr: false, watch: null },
    resolve: {
        alias: {
            "@web-falcor/falcor": resolve(repoRoot, "packages/falcor/src/index.ts"),
            "@web-falcor/render-passes": resolve(repoRoot, "packages/render-passes/src/index.ts"),
        },
    },
    logLevel: "warn",
    plugins: [dirListing({ root: repoRoot, allow: existsSync(resolve(repoRoot, "Falcor/media")) ? [realpathSync(resolve(repoRoot, "Falcor/media"))] : [] })],
});
await vite.listen();
const port = vite.config.server.port === 0 ? vite.httpServer.address().port : vite.config.server.port;
// EXTRA_QUERY (e.g. "dbg=scene/SceneCache") is appended for tests that read their own parameters.
const url = `http://127.0.0.1:${port}/tests/gpu/harness/index.html?filter=${encodeURIComponent(filter ?? "")}${process.env.EXTRA_QUERY ? `&${process.env.EXTRA_QUERY}` : ""}`;

// --disable-gpu-process-crash-limit: after a lost device main.ts recreates it; Chromium would refuse WebGPU after repeated GPU-process crashes.
const args = ["--enable-unsafe-webgpu", "--no-sandbox", "--disable-gpu-sandbox", "--ignore-gpu-blocklist", "--disable-gpu-process-crash-limit"];
if (useSwiftShader) {
    args.push("--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader");
} else {
    args.push("--enable-features=Vulkan");
}
// CHROME_ARGS (space-separated, e.g. "--js-flags=--expose-gc") for debugging runs.
if (process.env.CHROME_ARGS) args.push(...process.env.CHROME_ARGS.split(" ").filter(Boolean));

const browser = await chromium.launch({
    channel: "chromium",
    // Hardware Vulkan requires the headed GPU init path (under Xvfb); SwiftShader works headless.
    headless: useSwiftShader,
    ignoreDefaultArgs: ["--enable-features"],
    args,
});

const page = await browser.newPage();
page.on("console", (msg) => {
    if (msg.type() === "error" || process.env.FORWARD_ALL) console.error("[browser]", msg.text());
});
page.on("pageerror", (err) => console.error("[pageerror]", err.message));
// A crashed page never finishes: report and exit instead of waiting out the suite timeout.
page.on("crash", async () => {
    console.error("[crash] the page crashed (out of memory?)");
    await browser.close().catch(() => {});
    await vite.close().catch(() => {});
    process.exit(2);
});

// A test logging "#HEAPSNAPSHOT <file>" gets a heap snapshot written there (then logs "[heap] snapshot written").
{
    const cdp = await page.context().newCDPSession(page);
    page.on("console", async (msg) => {
        const m = msg.text().match(/^#HEAPSNAPSHOT (.*)/);
        if (!m) return;
        const { createWriteStream } = await import("node:fs");
        const out = createWriteStream(m[1]);
        const onChunk = ({ chunk }) => out.write(chunk);
        cdp.on("HeapProfiler.addHeapSnapshotChunk", onChunk);
        await cdp.send("HeapProfiler.takeHeapSnapshot", { reportProgress: false });
        cdp.off("HeapProfiler.addHeapSnapshotChunk", onChunk);
        out.end(() => console.error(`[heap] snapshot written to ${m[1]}`));
    });
}
// CPU_PROFILE=1: a test logging "#CPUPROFILE_START" / "#CPUPROFILE_STOP <label>" gets the top self-time functions printed.
if (process.env.CPU_PROFILE) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Profiler.enable");
    await cdp.send("Profiler.setSamplingInterval", { interval: 1000 });
    page.on("console", async (msg) => {
        const text = msg.text();
        if (text === "#CPUPROFILE_START") return void (await cdp.send("Profiler.start"));
        const m = text.match(/^#CPUPROFILE_STOP (.*)/);
        if (!m) return;
        const { profile } = await cdp.send("Profiler.stop");
        const dt = profile.timeDeltas.reduce((a, b) => a + b, 0) / Math.max(1, profile.samples.length);
        const self = new Map();
        const byId = new Map(profile.nodes.map((n) => [n.id, n]));
        for (const id of profile.samples) {
            const f = byId.get(id).callFrame;
            const key = `${f.functionName || "(anon)"} ${f.url.split("/").pop()}:${f.lineNumber + 1}`;
            self.set(key, (self.get(key) ?? 0) + dt);
        }
        const top = [...self].sort((a, b) => b[1] - a[1]).slice(0, 20);
        console.error(`[cpu] ${m[1]}: ${top.map(([k, v]) => `\n  ${(v / 1000).toFixed(0)}ms ${k}`).join("")}`);
    });
}
// HEAP_SAMPLING=1: sampled live allocations; a test logging "#HEAPPROFILE <label>" gets the top allocation sites printed.
if (process.env.HEAP_SAMPLING) {
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("HeapProfiler.enable");
    await cdp.send("HeapProfiler.startSampling", { samplingInterval: 64 * 1024 });
    page.on("console", async (msg) => {
        const m = msg.text().match(/^#HEAPPROFILE (.*)/);
        if (!m) return;
        const { profile } = await cdp.send("HeapProfiler.getSamplingProfile");
        const sites = new Map();
        const walk = (node, stack) => {
            const f = node.callFrame;
            const here = [...stack, `${f.functionName || "(anon)"} ${f.url.split("/").pop()}:${f.lineNumber + 1}`];
            if (node.selfSize) {
                const key = here.slice(-4).reverse().join(" < ");
                sites.set(key, (sites.get(key) ?? 0) + node.selfSize);
            }
            for (const c of node.children) walk(c, here);
        };
        walk(profile.head, []);
        const top = [...sites].sort((a, b) => b[1] - a[1]).slice(0, 15);
        console.error(`[heap] ${m[1]}: ${top.map(([k, v]) => `\n  ${(v / 1e6).toFixed(0)}MB ${k}`).join("")}`);
    });
}

await page.goto(url);
// Wall clock for the whole suite (grew past 10 min with ~170 tests; the per-test page keeps streaming results).
await page.waitForFunction(() => window.__done === true, undefined, { timeout: 55 * 60 * 1000 });

const fatal = await page.evaluate(() => window.__fatal);
const results = (await page.evaluate(() => window.__results)) ?? [];
const artifacts = (await page.evaluate(() => window.__artifacts)) ?? [];

await browser.close();
await vite.close();

// Write any PNG artifacts tests saved (via saveArtifact) to tests/gpu/out/.
if (artifacts.length > 0) {
    const { writeFileSync, mkdirSync } = await import("node:fs");
    const outDir = resolve(repoRoot, "tests/gpu/out");
    mkdirSync(outDir, { recursive: true });
    for (const a of artifacts) {
        const file = resolve(outDir, `${a.name}.png`);
        writeFileSync(file, Buffer.from(a.b64, "base64"));
        console.log(`🖼  artifact: ${file} (${a.width}x${a.height})`);
    }
}

if (fatal) {
    console.error(`FATAL: ${fatal}`);
    process.exit(2);
}

const pass = results.filter((r) => r.status === "pass").length;
const fail = results.filter((r) => r.status === "fail");
const skip = results.filter((r) => r.status === "skip").length;

for (const r of results) {
    const mark = r.status === "pass" ? "✓" : r.status === "skip" ? "→" : "✗";
    console.log(`${mark} ${r.name} (${r.ms.toFixed(1)}ms)${r.error ? `\n    ${r.error.split("\n")[0]}` : ""}`);
}
console.log(`\n${pass} passed, ${fail.length} failed, ${skip} skipped`);
for (const r of fail) console.error(`\nFAIL ${r.name}\n${r.error}`);
process.exit(fail.length > 0 ? 1 : 0);
