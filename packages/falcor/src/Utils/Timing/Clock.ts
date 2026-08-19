/**
 * Global clock mirroring Utils/Timing/Clock: real-time mode (framerate 0,
 * timer-driven with time scale) or FPS-simulation mode (fixed ticks per
 * frame — the mode every native image test drives via
 * `m.clock.framerate = N; m.clock.pause(); m.clock.frame = K`).
 * Times quantize to the native tick grid (14400 * 2^16 ticks/second).
 */

// 14400 is a common multiple of the supported frame rates; 2^16 gives headroom (Clock.cpp).
const kTicksPerSecond = 14400 * (1 << 16);

function timeFromFrame(frame: number, ticksPerFrame: number): number {
    return (frame * ticksPerFrame) / kTicksPerSecond;
}

function frameFromTime(seconds: number, ticksPerFrame: number): number {
    return Math.floor((seconds * kTicksPerSecond) / ticksPerFrame);
}

export class Clock {
    private fps = 0;
    private ticksPerFrame = 0;
    private frames = 0;
    private paused = false;
    private scale = 1;
    private now = 0;
    private delta = 0;
    private startTime = 0;
    private endTime = -1;
    private deferredTime: number | null = null;
    private deferredFrame: number | null = null;
    private lastRealTime: number;

    /** `nowSeconds` is injectable for tests (defaults to performance.now). */
    constructor(private readonly nowSeconds: () => number = () => performance.now() / 1000) {
        this.lastRealTime = this.nowSeconds();
    }

    private clampTime(seconds: number): number {
        const hi = this.endTime >= 0 ? this.endTime : Infinity;
        return Math.min(Math.max(seconds, this.startTime), hi);
    }

    private updateTimer(): number {
        const t = this.nowSeconds();
        const dt = t - this.lastRealTime;
        this.lastRealTime = t;
        return dt;
    }

    private update(t: number): void {
        this.delta = t - this.now;
        this.now = t;
    }

    isSimulatingFps(): boolean {
        return this.fps !== 0;
    }

    /** Advances one frame per call (or applies a deferred time/frame set). */
    tick(): this {
        if (this.deferredFrame !== null) this.setFrame(this.deferredFrame);
        else if (this.deferredTime !== null) this.setTime(this.deferredTime);
        else if (!this.paused) this.step();
        return this;
    }

    step(frames = 1): this {
        this.frames = Math.max(0, this.frames + frames);
        const dt = this.updateTimer();
        let t = this.isSimulatingFps() ? timeFromFrame(this.frames, this.ticksPerFrame) : dt * this.scale + this.now;
        t = this.clampTime(t);
        this.update(t);
        return this;
    }

    setTime(seconds: number, deferToNextTick = false): this {
        this.deferredTime = null;
        this.deferredFrame = null;
        seconds = this.clampTime(seconds);
        if (deferToNextTick) {
            this.deferredTime = seconds;
        } else {
            this.updateTimer();
            if (this.fps) {
                this.frames = frameFromTime(seconds, this.ticksPerFrame);
                seconds = timeFromFrame(this.frames, this.ticksPerFrame);
            }
            this.update(seconds);
        }
        return this;
    }

    setFrame(f: number, deferToNextTick = false): this {
        this.deferredTime = null;
        this.deferredFrame = null;
        if (deferToNextTick) {
            this.deferredFrame = f;
        } else {
            this.updateTimer();
            this.frames = f;
            if (this.fps) {
                this.update(this.clampTime(timeFromFrame(this.frames, this.ticksPerFrame)));
            }
        }
        return this;
    }

    setFramerate(fps: number): this {
        this.fps = fps;
        this.ticksPerFrame = 0;
        if (fps) {
            if (kTicksPerSecond % fps) console.warn("Clock.setFramerate: requested FPS can't be accurately represented. Expect rounding errors");
            this.ticksPerFrame = Math.floor(kTicksPerSecond / fps);
        }
        if (this.deferredFrame === null && this.deferredTime === null) this.setTime(this.now);
        return this;
    }

    setTimeScale(scale: number): this {
        this.scale = scale;
        return this;
    }

    setStartTime(t: number): void {
        this.startTime = Math.max(0, t);
    }

    setEndTime(t: number): void {
        this.endTime = t;
    }

    pause(): this {
        this.paused = true;
        return this;
    }

    play(): this {
        this.updateTimer();
        this.paused = false;
        return this;
    }

    /** Mirrors Clock::stop: rewind to start and pause. */
    stop(): this {
        this.setTime(0);
        this.paused = true;
        return this;
    }

    getTime(): number {
        return this.now;
    }

    getDelta(): number {
        return this.delta;
    }

    getFrame(): number {
        return this.frames;
    }

    getFramerate(): number {
        return this.fps;
    }

    getTimeScale(): number {
        return this.scale;
    }

    isPaused(): boolean {
        return this.paused;
    }

    /** Python-facing property surface (mirrors the native pybind names). */
    get time(): number {
        return this.getTime();
    }
    set time(t: number) {
        this.setTime(t);
    }
    get frame(): number {
        return this.getFrame();
    }
    set frame(f: number) {
        this.setFrame(f);
    }
    get framerate(): number {
        return this.fps;
    }
    set framerate(fps: number) {
        this.setFramerate(fps);
    }
    get timeScale(): number {
        return this.scale;
    }
    set timeScale(s: number) {
        this.setTimeScale(s);
    }
}
