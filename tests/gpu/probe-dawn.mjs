import { chromium } from "playwright";
const browser = await chromium.launch({ channel: "chromium", headless: true, ignoreDefaultArgs: ["--enable-features"],
  args: ["--enable-unsafe-webgpu","--no-sandbox","--enable-features=Vulkan","--ignore-gpu-blocklist","--disable-gpu-sandbox"] });
const page = await browser.newPage();
await page.goto("chrome://gpu", { waitUntil: "domcontentloaded" });
await page.waitForTimeout(2000);
const text = await page.evaluate(() => {
  const walk = (root) => {
    let out = root.textContent ?? "";
    for (const el of root.querySelectorAll("*")) if (el.shadowRoot) out += walk(el.shadowRoot);
    return out;
  };
  return walk(document);
});
const idx = text.indexOf("Dawn Info");
console.log(text.slice(idx, idx + 1200));
await browser.close();
