/** Mirrors Utils/Timing/FrameRate.h: 60-frame window of frame times (seconds). */

const kFrameWindow = 60;

export class FrameRate {
    private frameTimes = new Float64Array(kFrameWindow);
    private frameCount = 0;
    private last = 0;

    constructor(private readonly now: () => number = () => performance.now()) {
        this.reset();
    }

    reset(): void {
        this.frameCount = 0;
        this.last = this.now();
    }

    newFrame(): void {
        this.frameCount++;
        const t = this.now();
        this.frameTimes[this.frameCount % kFrameWindow] = (t - this.last) / 1000;
        this.last = t;
    }

    getAverageFrameTime(): number {
        const frames = Math.min(this.frameCount, kFrameWindow);
        if (frames === 0) return 0;
        let time = 0;
        for (let i = 0; i < frames; i++) time += this.frameTimes[i]!;
        return time / frames;
    }

    getLastFrameTime(): number {
        return this.frameTimes[this.frameCount % kFrameWindow]!;
    }

    getFrameCount(): number {
        return this.frameCount;
    }

    /** "60.0 FPS (16.7 ms/frame)[, VSync]". */
    getMsg(vsyncOn = false): string {
        const frameTime = this.getAverageFrameTime();
        return `${(1 / frameTime).toFixed(1)} FPS (${(frameTime * 1000).toFixed(1)} ms/frame)${vsyncOn ? ", VSync" : ""}`;
    }
}
