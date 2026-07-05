import { chromium } from "playwright";
import { createServer } from "node:http";
const server = createServer((_q, s) => { s.setHeader("content-type", "text/html"); s.end("<!doctype html>"); });
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const browser = await chromium.launch({ channel: "chromium", headless: false, ignoreDefaultArgs: ["--enable-features"],
  args: ["--enable-unsafe-webgpu","--enable-webgpu-developer-features","--enable-dawn-features=allow_unsafe_apis","--no-sandbox","--enable-features=Vulkan","--ignore-gpu-blocklist","--disable-gpu-sandbox"] });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`);
const r = await page.evaluate(async () => {
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  const hasF16 = a.features.has("shader-f16");
  let deviceOk = false;
  if (hasF16) {
    const d = await a.requestDevice({ requiredFeatures: ["shader-f16"] });
    // compile a trivial f16 shader
    const mod = d.createShaderModule({ code: "enable f16; @compute @workgroup_size(1) fn main() { var x: f16 = 1.0h; _ = x; }" });
    const info = await mod.getCompilationInfo();
    deviceOk = !info.messages.some((m) => m.type === "error");
  }
  return { hasF16, deviceOk, all: [...a.features].filter((f) => f.includes("f16") || f.includes("float")) };
});
console.log(JSON.stringify(r));
await browser.close(); server.close();
