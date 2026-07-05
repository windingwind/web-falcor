import { chromium } from "playwright";
import { createServer } from "node:http";
const server = createServer((_q, s) => { s.setHeader("content-type", "text/html"); s.end("<!doctype html>"); });
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const combos = [
  { name: "angle-vulkan-shared", args: ["--enable-unsafe-webgpu","--no-sandbox","--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan","--ignore-gpu-blocklist","--disable-vulkan-surface"] },
  { name: "use-vulkan-native", args: ["--enable-unsafe-webgpu","--no-sandbox","--use-vulkan=native","--enable-features=Vulkan","--ignore-gpu-blocklist","--disable-vulkan-surface","--disable-gpu-sandbox"] },
  { name: "force-hw-adapter", args: ["--enable-unsafe-webgpu","--no-sandbox","--enable-features=Vulkan","--ignore-gpu-blocklist","--use-webgpu-adapter=default","--disable-software-rasterizer"] },
];
for (const c of combos) {
  const browser = await chromium.launch({ channel: "chromium", headless: true, ignoreDefaultArgs: ["--enable-features"], args: c.args });
  try {
    const page = await browser.newPage();
    await page.goto(`http://127.0.0.1:${port}/`);
    const r = await page.evaluate(async () => {
      if (!("gpu" in navigator)) return { ok: false, reason: "no navigator.gpu" };
      const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!a) return { ok: false, reason: "no adapter" };
      const i = a.info ?? {};
      return { ok: true, vendor: i.vendor, arch: i.architecture, desc: i.description, sub: a.isFallbackAdapter };
    });
    console.log(`[${c.name}]`, JSON.stringify(r));
    if (r.ok && r.arch !== "swiftshader") { console.log("HW_WINNER=" + JSON.stringify(c.args)); break; }
  } finally { await browser.close(); }
}
server.close();
