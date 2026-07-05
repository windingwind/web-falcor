#!/usr/bin/env node
/**
 * GPU test driver: serves the workspace with Vite, launches Chromium
 * (hardware WebGPU via Vulkan; requires an X display — run under `xvfb-run -a`),
 * executes the browser-side runner, reports results.
 *
 * Usage: xvfb-run -a node tests/gpu/harness/run.mjs [--swiftshader]
 */

import { createServer } from "vite";
import { chromium } from "playwright";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const useSwiftShader = process.argv.includes("--swiftshader");

const vite = await createServer({
    root: repoRoot,
    server: { port: 0, host: "127.0.0.1" },
    resolve: {
        alias: { "@web-falcor/falcor": resolve(repoRoot, "packages/falcor/src/index.ts") },
    },
    logLevel: "warn",
});
await vite.listen();
const port = vite.config.server.port === 0 ? vite.httpServer.address().port : vite.config.server.port;
const url = `http://127.0.0.1:${port}/tests/gpu/harness/index.html`;

const args = ["--enable-unsafe-webgpu", "--no-sandbox", "--disable-gpu-sandbox", "--ignore-gpu-blocklist"];
if (useSwiftShader) {
    args.push("--enable-features=Vulkan", "--use-webgpu-adapter=swiftshader");
} else {
    args.push("--enable-features=Vulkan");
}

const browser = await chromium.launch({
    channel: "chromium",
    // Hardware Vulkan requires the headed GPU init path (under Xvfb); SwiftShader works headless.
    headless: useSwiftShader,
    ignoreDefaultArgs: ["--enable-features"],
    args,
});

const page = await browser.newPage();
page.on("console", (msg) => {
    if (msg.type() === "error") console.error("[browser]", msg.text());
});
page.on("pageerror", (err) => console.error("[pageerror]", err.message));

await page.goto(url);
await page.waitForFunction(() => window.__done === true, undefined, { timeout: 10 * 60 * 1000 });

const fatal = await page.evaluate(() => window.__fatal);
const results = (await page.evaluate(() => window.__results)) ?? [];

await browser.close();
await vite.close();

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
