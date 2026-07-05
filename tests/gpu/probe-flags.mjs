// Try flag combinations until headless Chromium exposes a WebGPU adapter.
// Serves over localhost (WebGPU needs a secure context).
import { chromium } from "playwright";
import { createServer } from "node:http";

const server = createServer((_req, res) => {
    res.setHeader("content-type", "text/html");
    res.end("<!doctype html><title>probe</title>");
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

const base = ["--enable-unsafe-webgpu", "--no-sandbox"];
const combos = [
    { name: "vulkan+angle+nogpusandbox", args: [...base, "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface", "--disable-gpu-sandbox", "--ignore-gpu-blocklist"] },
    { name: "vulkan+angle", args: [...base, "--enable-features=Vulkan", "--use-angle=vulkan", "--disable-vulkan-surface", "--ignore-gpu-blocklist"] },
    { name: "vulkan only", args: [...base, "--enable-features=Vulkan", "--ignore-gpu-blocklist"] },
    { name: "defaults+unsafe", args: [...base] },
    { name: "swiftshader", args: [...base, "--use-webgpu-adapter=swiftshader"] },
];

async function probe(args, dumpGpu = false) {
    const browser = await chromium.launch({
        channel: "chromium",
        headless: true,
        ignoreDefaultArgs: ["--enable-features"],
        args,
    });
    try {
        const page = await browser.newPage();
        await page.goto(`http://127.0.0.1:${port}/`);
        const result = await page.evaluate(async () => {
            if (!("gpu" in navigator)) return { ok: false, reason: "navigator.gpu missing" };
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) return { ok: false, reason: "no adapter" };
            const info = adapter.info ?? {};
            return { ok: true, vendor: info.vendor, arch: info.architecture, desc: info.description };
        });
        if (dumpGpu && !result.ok) {
            const gpuPage = await browser.newPage();
            await gpuPage.goto("chrome://gpu", { waitUntil: "domcontentloaded" });
            await gpuPage.waitForTimeout(1500);
            const text = await gpuPage.evaluate(() => {
                const el = document.querySelector("info-view");
                return (el?.shadowRoot?.textContent ?? document.body.innerText ?? "").slice(0, 20000);
            });
            const interesting = text
                .split("\n")
                .filter((l) => /webgpu|vulkan|angle|gl_renderer|driver|skia|problem|disabled|blocklist/i.test(l))
                .slice(0, 30);
            console.log("  chrome://gpu says:");
            for (const l of interesting) console.log("   |", l.trim().slice(0, 160));
        }
        return result;
    } finally {
        await browser.close();
    }
}

for (const combo of combos) {
    try {
        const result = await probe(combo.args, combo === combos[0]);
        console.log(`[${combo.name}]`, JSON.stringify(result));
        if (result.ok) {
            console.log("WINNER_ARGS=" + JSON.stringify(combo.args));
            server.close();
            process.exit(0);
        }
    } catch (err) {
        console.log(`[${combo.name}] error: ${err.message.split("\n")[0]}`);
    }
}
server.close();
process.exit(1);
