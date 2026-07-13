/**
 * Canvas video recording (the Mogwai VideoCapture role; native encodes via
 * ffmpeg/VideoEncoder). Web divergence (docs §9): MediaRecorder over
 * canvas.captureStream — container/codec is what the browser offers
 * (WebM VP9/VP8), constant quality only.
 */

import { RuntimeError } from "../../Core/Error.js";

const kPreferredTypes = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];

interface CanvasCaptureTrack extends MediaStreamTrack {
    requestFrame(): void;
}

export class VideoRecorder {
    private recorder: MediaRecorder | null = null;
    private track: CanvasCaptureTrack | null = null;
    private chunks: Blob[] = [];
    private stopped: Promise<Blob> | null = null;

    get recording(): boolean {
        return this.recorder !== null && this.recorder.state === "recording";
    }

    /** Starts recording; add frames with captureFrame() after each present
     *  (native VideoCapture encodes per rendered frame, not wall-clock). */
    start(canvas: HTMLCanvasElement, mimeType?: string): void {
        if (this.recorder) throw new RuntimeError("VideoRecorder: already recording");
        const type = mimeType ?? kPreferredTypes.find((t) => MediaRecorder.isTypeSupported(t));
        if (!type) throw new RuntimeError("VideoRecorder: no supported video mime type");
        const stream = canvas.captureStream(0);
        this.track = stream.getVideoTracks()[0] as CanvasCaptureTrack;
        this.chunks = [];
        this.recorder = new MediaRecorder(stream, { mimeType: type });
        this.recorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.chunks.push(e.data);
        };
        this.stopped = new Promise<Blob>((resolve, reject) => {
            this.recorder!.onstop = () => resolve(new Blob(this.chunks, { type }));
            this.recorder!.onerror = (e) => reject(new RuntimeError(`VideoRecorder: ${(e as ErrorEvent).error ?? "recording failed"}`));
        });
        this.recorder.start();
    }

    /** Pushes the canvas's current contents as the next video frame. */
    captureFrame(): void {
        this.track?.requestFrame();
    }

    /** Stops recording and returns the encoded video. */
    async stop(): Promise<Blob> {
        if (!this.recorder || !this.stopped) throw new RuntimeError("VideoRecorder: not recording");
        this.recorder.stop();
        const blob = await this.stopped;
        this.recorder = null;
        this.track = null;
        this.stopped = null;
        return blob;
    }
}
