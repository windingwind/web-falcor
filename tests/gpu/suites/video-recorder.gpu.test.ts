/**
 * VideoRecorder (Mogwai VideoCapture role): records an animating canvas via
 * MediaRecorder and yields a non-empty decodable WebM blob.
 */

import { VideoRecorder } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("VideoRecorder.capturesAnimatedCanvas", async () => {
    const canvas = document.createElement("canvas");
    canvas.width = 128;
    canvas.height = 128;
    document.body.appendChild(canvas);
    const c2d = canvas.getContext("2d")!;

    const rec = new VideoRecorder();
    rec.start(canvas);
    expectEq(rec.recording, true, "recording started");

    // Animate ~0.5 s so several frames land in the stream.
    const t0 = performance.now();
    while (performance.now() - t0 < 500) {
        c2d.fillStyle = `hsl(${(performance.now() - t0) * 0.7}, 90%, 50%)`;
        c2d.fillRect(0, 0, 128, 128);
        rec.captureFrame();
        await new Promise((r) => requestAnimationFrame(r));
    }

    const blob = await rec.stop();
    console.error(`# video: ${blob.size} bytes, type=${blob.type}`);
    expectEq(rec.recording, false, "recording stopped");
    expectEq(blob.type.startsWith("video/webm"), true, `webm container (${blob.type})`);
    expectEq(blob.size > 1000, true, `encoded data present (${blob.size} bytes)`);
    canvas.remove();
});
