import { chromium } from "playwright";
import { createServer } from "node:http";
const server = createServer((_q, s) => { s.setHeader("content-type", "text/html"); s.end("<!doctype html>"); });
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const port = server.address().port;
const browser = await chromium.launch({ channel: "chromium", headless: false, ignoreDefaultArgs: ["--enable-features"],
  args: ["--enable-unsafe-webgpu","--no-sandbox","--enable-features=Vulkan","--ignore-gpu-blocklist","--disable-gpu-sandbox"] });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${port}/`);
const r = await page.evaluate(async () => {
  if (!("gpu" in navigator)) return { ok: false, reason: "no navigator.gpu" };
  const a = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!a) return { ok: false, reason: "no adapter" };
  const i = a.info ?? {};
  return { ok: true, vendor: i.vendor, arch: i.architecture, desc: i.description, features: [...a.features].sort() };
});
console.log(JSON.stringify(r, null, 1));
await browser.close(); server.close();
