// Probe: does headless Chromium expose a WebGPU adapter on this host?
// Note: WebGPU requires a secure context -> serve over http://localhost.
import { chromium } from "playwright";
import { createServer } from "node:http";

const server = createServer((_req, res) => {
    res.setHeader("content-type", "text/html");
    res.end("<!doctype html><title>probe</title>");
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

const browser = await chromium.launch({
    channel: "chromium",
    headless: true,
    ignoreDefaultArgs: ["--enable-features"],
    args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-angle=vulkan",
        "--disable-vulkan-surface",
        "--no-sandbox",
    ],
});
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`);
const result = await page.evaluate(async () => {
    if (!("gpu" in navigator)) return { ok: false, reason: "navigator.gpu missing", secure: window.isSecureContext };
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) return { ok: false, reason: "no adapter" };
    const info = adapter.info ?? {};
    return {
        ok: true,
        vendor: info.vendor,
        architecture: info.architecture,
        device: info.device,
        description: info.description,
        features: [...adapter.features].sort(),
        limits: {
            maxBufferSize: adapter.limits.maxBufferSize,
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
        },
    };
});
console.log(JSON.stringify(result, null, 2));
await browser.close();
server.close();
process.exit(result.ok ? 0 : 1);
