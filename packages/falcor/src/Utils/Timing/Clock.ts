/**
 * Global clock mirroring Utils/Timing/Clock: real-time mode (framerate 0,
 * timer-driven with time scale) or FPS-simulation mode (fixed ticks per
 * frame — the mode every native image test drives via
 * `m.clock.framerate = N; m.clock.pause(); m.clock.frame = K`).
 * Times quantize to the native tick grid (14400 * 2^16 ticks/second).
 */

// 14400 is a common multiple of the supported frame rates; 2^16 gives headroom (Clock.cpp).
import { ScriptWriter } from "../Scripting/ScriptWriter.js";
import type { UIWidgets } from "../../RenderGraph/UIWidgets.js";

/** fpsDropdown's common rates (0 = simulation disabled). */
const kCommonFps = [0, 24, 25, 30, 48, 50, 60, 75, 90, 120, 144, 200, 240, 360, 480];
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
    private mStartTime = 0;
    private mEndTime = -1;
    /** Mirrors Clock's exit time/frame (0 = none): the app quits once reached. */
    private mExitTime = 0;
    private mExitFrame = 0;
    private deferredTime: number | null = null;
    private deferredFrame: number | null = null;
    private lastRealTime: number;

    /** `nowSeconds` is injectable for tests (defaults to performance.now). */
    constructor(private readonly nowSeconds: () => number = () => performance.now() / 1000) {
        this.lastRealTime = this.nowSeconds();
    }

    private clampTime(seconds: number): number {
        const hi = this.mEndTime >= 0 ? this.mEndTime : Infinity;
        return Math.min(Math.max(seconds, this.mStartTime), hi);
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

    /** Mirrors Clock::setStartTime: <= 0 resets to 0; otherwise it must precede a set end time. */
    setStartTime(time: number): boolean {
        if (time <= 0) {
            this.mStartTime = 0;
            return true;
        }
        if (this.mEndTime < 0 || time < this.mEndTime) {
            this.mStartTime = time;
            return true;
        }
        return false;
    }
    getStartTime(): number {
        return this.mStartTime;
    }

    /** Mirrors Clock::setEndTime: < 0 disables the loop; otherwise it must follow a set start time. */
    setEndTime(time: number): boolean {
        if (time < 0 || this.mStartTime <= 0 || time > this.mStartTime) {
            this.mEndTime = time;
            return true;
        }
        return false;
    }
    getEndTime(): number {
        return this.mEndTime;
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

    /** Mirrors Clock::setExitTime (seconds; 0 disables). */
    setExitTime(seconds: number): this {
        this.mExitTime = seconds;
        return this;
    }
    getExitTime(): number {
        return this.mExitTime;
    }
    /** Mirrors Clock::setExitFrame (0 disables). */
    setExitFrame(frame: number): this {
        this.mExitFrame = frame;
        return this;
    }
    getExitFrame(): number {
        return this.mExitFrame;
    }
    /** Mirrors Clock::shouldExit. */
    shouldExit(): boolean {
        return (this.mExitTime > 0 && this.now >= this.mExitTime) || (this.mExitFrame > 0 && this.frames >= this.mExitFrame);
    }

    /** Mirrors Clock::renderUI: time and scale, rewind/stop/play-pause (frame steps while paused), frame-rate simulation. */
    renderUI(w: UIWidgets): void {
        const time = this.getTime();
        w.slider("Time", time, 0, Math.max(60, Math.ceil(time * 2)), 0.001, (t) => this.setTime(t));
        if (!this.isSimulatingFps()) w.slider("Scale", this.getTimeScale(), 0, 10, 0.01, (v) => this.setTimeScale(v));
        const showStep = this.isPaused() && this.isSimulatingFps();
        w.button("Rewind", () => this.setTime(0));
        if (showStep) w.button("Prev Frame", () => this.step(-1));
        w.button("Stop", () => this.stop());
        w.button(this.isPaused() ? "Play" : "Pause", () => (this.isPaused() ? this.play() : this.pause()));
        if (showStep) w.button("Next Frame", () => this.step());
        w.text("Framerate Simulation (time advances by 1/FPS per frame)");
        const fps = this.getFramerate();
        const names = [...kCommonFps.map((f) => (f === 0 ? "Disabled" : String(f))), "Custom"];
        const current = kCommonFps.includes(fps) ? (fps === 0 ? "Disabled" : String(fps)) : "Custom";
        w.dropdown("FPS", names, current, (v) => this.setFramerate(v === "Disabled" ? 0 : v === "Custom" ? Math.max(1, fps) : Number(v)));
        if (current === "Custom") w.slider("Custom FPS", fps, 1, 1000, 1, (v) => this.setFramerate(Math.round(v)));
        if (this.isSimulatingFps()) w.slider("Frame ID", this.getFrame(), 0, Math.max(1000, this.getFrame() * 2), 1, (f) => this.setFrame(Math.round(f)));
    }

    /** Mirrors Clock::getScript: the settings as script lines on `variable`. */
    getScript(variable: string): string {
        let s = ScriptWriter.makeSetProperty(variable, "time", 0);
        s += ScriptWriter.makeSetProperty(variable, "framerate", this.fps);
        if (this.mExitTime) s += ScriptWriter.makeSetProperty(variable, "exitTime", this.mExitTime);
        if (this.mExitFrame) s += ScriptWriter.makeSetProperty(variable, "exitFrame", this.mExitFrame);
        s += "# If framerate is not zero, you can use the frame property to set the start frame\n";
        s += `# ${ScriptWriter.makeSetProperty(variable, "frame", 0)}`;
        if (this.paused) s += ScriptWriter.makeMemberFunc(variable, "pause");
        return s;
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
    get startTime(): number {
        return this.mStartTime;
    }
    set startTime(t: number) {
        this.setStartTime(t);
    }
    get endTime(): number {
        return this.mEndTime;
    }
    set endTime(t: number) {
        this.setEndTime(t);
    }
    get exitTime(): number {
        return this.mExitTime;
    }
    set exitTime(t: number) {
        this.setExitTime(t);
    }
    get exitFrame(): number {
        return this.mExitFrame;
    }
    set exitFrame(f: number) {
        this.setExitFrame(f);
    }
}
