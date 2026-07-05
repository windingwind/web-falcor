import { chromium } from "playwright";

const browser = await chromium.launch({
    channel: "chromium",
    headless: true,
    ignoreDefaultArgs: ["--enable-features"],
    args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan,WebGPUService",
        "--enable-blink-features=WebGPU",
        "--use-angle=vulkan",
        "--disable-vulkan-surface",
        "--ignore-gpu-blocklist",
        "--no-sandbox",
    ],
});
console.log("version:", browser.version());
const page = await browser.newPage();
console.log("gpu in navigator:", await page.evaluate(() => "gpu" in navigator));
console.log("isSecureContext:", await page.evaluate(() => window.isSecureContext));
console.log("UA:", await page.evaluate(() => navigator.userAgent));

const gpuPage = await browser.newPage();
await gpuPage.goto("chrome://gpu");
const text = await gpuPage.evaluate(() => document.body.innerText);
const lines = text.split("\n").filter((l) => /webgpu|vulkan|angle|gl renderer|driver/i.test(l)).slice(0, 25);
console.log(lines.join("\n"));
await browser.close();
