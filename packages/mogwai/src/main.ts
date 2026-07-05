/**
 * Mogwai (web) entry point. Currently a device bring-up sanity check:
 * initializes the Device, configures the swapchain, and clears the canvas.
 */

import { Device, Logger } from "@web-falcor/falcor";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const status = document.getElementById("status") as HTMLDivElement;

async function main() {
    const device = await Device.create();
    const features = device.getSupportedFeatures();

    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get webgpu canvas context");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device.gpuDevice, format });

    status.textContent =
        `web-falcor device OK\n` +
        `adapter: ${device.adapter.info?.vendor ?? "?"} ${device.adapter.info?.architecture ?? ""}\n` +
        `features: ${JSON.stringify(features, null, 1)}`;

    function frame(t: number) {
        const encoder = device.gpuDevice.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: context!.getCurrentTexture().createView(),
                    clearValue: { r: 0.1, g: 0.2, b: 0.05 + 0.05 * Math.sin(t / 500), a: 1 },
                    loadOp: "clear",
                    storeOp: "store",
                },
            ],
        });
        pass.end();
        device.queue.submit([encoder.finish()]);
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

main().catch((err) => {
    Logger.error(String(err));
    status.textContent = `FAILED: ${err.message ?? err}`;
});
