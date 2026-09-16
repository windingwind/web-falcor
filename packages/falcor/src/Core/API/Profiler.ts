/**
 * Profiler mirroring Utils/Timing/Profiler.h: nested named events with CPU
 * time (performance.now) and GPU time (WebGPU pass `timestampWrites`
 * attributed to every event active when the pass began, so parents include
 * their children like native GpuTimer spans), EMA averages, a 512-frame
 * history for stats, pause, and multi-frame captures (JSON lanes).
 * Web divergences (docs §9): GPU readback is async (event times land a
 * frame or two late; frames beyond the in-flight ring are dropped instead of
 * blocking on a fence) and the profiler is enabled by default.
 */

import type { Device } from "./Device.js";
import { Logger } from "../../Utils/Logger.js";

const kSigma = 0.98;
const kMaxHistorySize = 512;
const kMaxTimestamps = 512;
const kReadbackRing = 4;

export interface ProfilerStats {
    min: number;
    max: number;
    mean: number;
    stdDev: number;
}

/** Mirrors Profiler::Stats::compute (double accumulation, population variance). */
export function computeStats(data: ArrayLike<number>, len = data.length): ProfilerStats {
    if (len === 0) return { min: 0, max: 0, mean: 0, stdDev: 0 };
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sum2 = 0;
    for (let i = 0; i < len; i++) {
        const v = data[i]!;
        min = Math.min(min, v);
        max = Math.max(max, v);
        sum += v;
        sum2 += v * v;
    }
    const mean = sum / len;
    const variance = len > 1 ? Math.max(sum2 / len - mean * mean, 0) : 0;
    return { min, max, mean: Math.fround(mean), stdDev: Math.fround(Math.sqrt(variance)) };
}

/** Mirrors Profiler::Event (times in milliseconds). */
export class ProfilerEvent {
    cpuTime = 0;
    gpuTime = 0;
    /** EMA averages; negative until the first measurement (native convention). */
    cpuTimeAverage = -1;
    gpuTimeAverage = -1;
    private cpuHistory = new Float32Array(kMaxHistorySize);
    private gpuHistory = new Float32Array(kMaxHistorySize);
    private historyWriteIndex = 0;
    private historySize = 0;
    // Current-frame accumulation.
    private triggered = 0;
    private cpuStart = 0;
    private cpuTotal = 0;
    private valid = false;

    constructor(readonly name: string) {}

    /** Nesting depth (0 for top-level events). */
    get level(): number {
        return Math.max((this.name.match(/\//g) ?? []).length, 1) - 1;
    }

    /** Last path component. */
    get shortName(): string {
        return this.name.slice(this.name.lastIndexOf("/") + 1);
    }

    computeCpuTimeStats(): ProfilerStats {
        return computeStats(this.cpuHistory, this.historySize);
    }
    computeGpuTimeStats(): ProfilerStats {
        return computeStats(this.gpuHistory, this.historySize);
    }
    resetStats(): void {
        this.historyWriteIndex = 0;
        this.historySize = 0;
    }

    /** @internal */
    start(now: number): void {
        if (++this.triggered > 1) {
            Logger.warning(`Profiler event '${this.name}' was triggered while it is already running. Nesting profiler events with the same name is disallowed and you should probably fix that. Ignoring the new call.`);
            return;
        }
        this.cpuStart = now;
        this.valid = false;
    }
    /** @internal */
    end(now: number): void {
        if (--this.triggered !== 0) return;
        this.cpuTotal += now - this.cpuStart;
        this.valid = true;
    }
    /** @internal Takes this frame's CPU total (null when nothing completed). */
    takeFrame(): number | null {
        const total = this.valid ? this.cpuTotal : null;
        this.cpuTotal = 0;
        this.triggered = 0;
        return total;
    }
    /** @internal Mirrors Event::endFrame's measurement update. */
    commit(cpuTime: number, gpuTime: number): void {
        this.cpuTime = cpuTime;
        this.gpuTime = gpuTime;
        this.cpuTimeAverage = this.cpuTimeAverage < 0 ? cpuTime : kSigma * this.cpuTimeAverage + (1 - kSigma) * cpuTime;
        this.gpuTimeAverage = this.gpuTimeAverage < 0 ? gpuTime : kSigma * this.gpuTimeAverage + (1 - kSigma) * gpuTime;
        this.cpuHistory[this.historyWriteIndex] = cpuTime;
        this.gpuHistory[this.historyWriteIndex] = gpuTime;
        this.historyWriteIndex = (this.historyWriteIndex + 1) % kMaxHistorySize;
        this.historySize = Math.min(this.historySize + 1, kMaxHistorySize);
    }
}

export interface ProfilerCaptureLane {
    name: string;
    stats: ProfilerStats;
    records: number[];
}

/** Mirrors Profiler::Capture: per-event cpu_time/gpu_time lanes over the captured frames. */
export class ProfilerCapture {
    frameCount = 0;
    lanes: ProfilerCaptureLane[] = [];
    private events: ProfilerEvent[] = [];

    /** @internal First call fixes the event set (no data yet, like native); later calls record. */
    captureEvents(events: readonly ProfilerEvent[]): void {
        if (events.length === 0) return;
        if (this.events.length === 0) {
            this.events = [...events];
            this.lanes = this.events.flatMap((e) => [
                { name: `${e.name}/cpu_time`, stats: computeStats([]), records: [] },
                { name: `${e.name}/gpu_time`, stats: computeStats([]), records: [] },
            ]);
            return;
        }
        this.events.forEach((e, i) => {
            this.lanes[i * 2]!.records.push(e.cpuTime);
            this.lanes[i * 2 + 1]!.records.push(e.gpuTime);
        });
        this.frameCount++;
    }
    /** @internal */
    finalize(): void {
        for (const lane of this.lanes) lane.stats = computeStats(lane.records);
    }

    /** Native toPython(Capture) layout: {frame_count, events: {name: {name, stats: {min,max,mean,std_dev}, records}}}. */
    toJson(): { frame_count: number; events: Record<string, { name: string; stats: Record<string, number>; records: number[] }> } {
        const events: Record<string, { name: string; stats: Record<string, number>; records: number[] }> = {};
        for (const lane of this.lanes) {
            events[lane.name] = { name: lane.name, stats: statsToJson(lane.stats), records: lane.records };
        }
        return { frame_count: this.frameCount, events };
    }
    toJsonString(): string {
        return JSON.stringify(this.toJson(), null, 2);
    }
}

function statsToJson(s: ProfilerStats): Record<string, number> {
    return { min: s.min, max: s.max, mean: s.mean, std_dev: s.stdDev };
}

interface TimestampEntry {
    /** Full names of every event active when the pass began. */
    names: string[];
    begin: number;
    end: number;
}

interface FrameRecord {
    events: ProfilerEvent[];
    cpuTotals: (number | null)[];
    entries: TimestampEntry[];
}

export class Profiler {
    private querySet: GPUQuerySet | null = null;
    private resolveBuffer: GPUBuffer | null = null;
    private resultBuffers: { buffer: GPUBuffer; busy: boolean }[] = [];
    private nextIndex = 0;
    private entries: TimestampEntry[] = [];

    private events = new Map<string, ProfilerEvent>();
    private currentFrameEvents: ProfilerEvent[] = [];
    private lastFrameEvents: ProfilerEvent[] = [];
    private activeNames: string[] = [];
    private frameIndex = 0;
    private pendingReset = false;
    private capture: ProfilerCapture | null = null;

    /** Web default: enabled (native starts disabled and toggles with the P key). */
    enabled = true;
    paused = false;

    constructor(
        private readonly device: Device | null,
        private readonly now: () => number = () => performance.now(),
    ) {
        if (device?.hasFeature("timestamp-query")) {
            this.querySet = device.gpuDevice.createQuerySet({ type: "timestamp", count: kMaxTimestamps });
            this.resolveBuffer = device.gpuDevice.createBuffer({ size: kMaxTimestamps * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
            for (let i = 0; i < kReadbackRing; i++) {
                this.resultBuffers.push({ buffer: device.gpuDevice.createBuffer({ size: kMaxTimestamps * 8, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }), busy: false });
            }
        }
    }

    /** True when GPU timestamps are available. */
    get available(): boolean {
        return this.querySet !== null;
    }

    isEnabled(): boolean { return this.enabled; }
    setEnabled(v: boolean): void { this.enabled = v; }
    isPaused(): boolean { return this.paused; }
    setPaused(v: boolean): void { this.paused = v; }
    getDevice(): Device | null { return this.device; }

    /** Mirrors Profiler::startEvent (names nest with '/'). */
    startEvent(name: string): void {
        if (!this.enabled) return;
        if (name.includes("/")) {
            Logger.warning("Profiler event names must not contain '/'. Ignoring this profiler event.");
            return;
        }
        const full = `${this.activeNames[this.activeNames.length - 1] ?? ""}/${name}`;
        this.activeNames.push(full);
        const event = this.getEvent(full);
        if (!this.paused) event.start(this.now());
        if (!this.currentFrameEvents.includes(event)) this.currentFrameEvents.push(event);
    }

    /** Mirrors Profiler::endEvent. */
    endEvent(name: string): void {
        if (!this.enabled || name.includes("/")) return;
        const full = this.activeNames.pop();
        if (full === undefined) return;
        if (!this.paused) this.getEvent(full).end(this.now());
    }

    /** Runs `fn` inside an event (FALCOR_PROFILE scope). */
    scope<T>(name: string, fn: () => T): T {
        this.startEvent(name);
        try {
            return fn();
        } finally {
            this.endEvent(name);
        }
    }

    /** Mirrors Profiler::getEvent (creates on first use). */
    getEvent(name: string): ProfilerEvent {
        let e = this.events.get(name);
        if (!e) {
            e = new ProfilerEvent(name);
            this.events.set(name, e);
        }
        return e;
    }
    findEvent(name: string): ProfilerEvent | undefined {
        return this.events.get(name);
    }
    /** Events of the last completed frame, in start order (tree order). */
    getEvents(): readonly ProfilerEvent[] {
        return this.lastFrameEvents;
    }

    /** Legacy view: GPU ms per graph pass (children of the graph event), else per top-level event. */
    getStats(): Map<string, number> {
        const stats = new Map<string, number>();
        const passes = this.lastFrameEvents.filter((e) => e.level === 1);
        for (const e of passes.length > 0 ? passes : this.lastFrameEvents.filter((e) => e.level === 0)) {
            stats.set(e.shortName, (stats.get(e.shortName) ?? 0) + e.gpuTime);
        }
        return stats;
    }

    /** Timestamp pair for a GPU pass begun now; the contexts attach it to the pass descriptor. */
    passTimestampWrites(): GPUComputePassTimestampWrites | undefined {
        if (!this.querySet || !this.enabled || this.paused || this.activeNames.length === 0 || this.nextIndex + 2 > kMaxTimestamps) return undefined;
        const begin = this.nextIndex;
        this.nextIndex += 2;
        this.entries.push({ names: [...this.activeNames], begin, end: begin + 1 });
        return { querySet: this.querySet, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: begin + 1 };
    }

    /**
     * Mirrors Profiler::endFrame: closes the frame's measurements (GPU times
     * land asynchronously; the frame commits once its timestamps are read).
     */
    endFrame(encoder?: GPUCommandEncoder | null): void {
        if (this.paused) {
            this.nextIndex = 0;
            this.entries = [];
            return;
        }
        const frame: FrameRecord = {
            events: this.currentFrameEvents,
            cpuTotals: this.currentFrameEvents.map((e) => e.takeFrame()),
            entries: this.entries,
        };
        const count = this.nextIndex;
        this.entries = [];
        this.nextIndex = 0;
        this.lastFrameEvents = this.currentFrameEvents;
        this.currentFrameEvents = [];
        this.frameIndex++;

        const slot = this.resultBuffers.find((s) => !s.busy);
        if (this.querySet && encoder && count > 0 && slot) {
            slot.busy = true;
            encoder.resolveQuerySet(this.querySet, 0, count, this.resolveBuffer!, 0);
            encoder.copyBufferToBuffer(this.resolveBuffer!, 0, slot.buffer, 0, count * 8);
            void this.device!.gpuDevice.queue.onSubmittedWorkDone().then(async () => {
                await slot.buffer.mapAsync(GPUMapMode.READ, 0, count * 8);
                const times = new BigUint64Array(slot.buffer.getMappedRange(0, count * 8).slice(0));
                slot.buffer.unmap();
                slot.busy = false;
                const gpu = new Map<string, number>();
                for (const e of frame.entries) {
                    const ms = Number(times[e.end]! - times[e.begin]!) / 1e6;
                    for (const n of e.names) gpu.set(n, (gpu.get(n) ?? 0) + ms);
                }
                this.commitFrame(frame, gpu);
            });
        } else if (!this.querySet || count === 0) {
            this.commitFrame(frame, new Map());
        }
        // else: readback ring full — this frame's measurements are dropped (native would block on a fence).
    }

    private commitFrame(frame: FrameRecord, gpu: Map<string, number>): void {
        frame.events.forEach((e, i) => {
            const cpu = frame.cpuTotals[i];
            if (cpu == null) return; // no completed measurement (native: !frameData.valid)
            e.commit(cpu, gpu.get(e.name) ?? 0);
        });
        this.capture?.captureEvents(frame.events);
        if (this.pendingReset) {
            for (const e of frame.events) e.resetStats();
            this.pendingReset = false;
        }
    }

    /** Mirrors Profiler::resetStats (applies at the next committed frame). */
    resetStats(): void {
        this.pendingReset = true;
    }

    startCapture(_reservedFrames = 1024): void {
        this.enabled = true;
        this.capture = new ProfilerCapture();
    }
    endCapture(): ProfilerCapture | null {
        const c = this.capture;
        this.capture = null;
        c?.finalize();
        return c;
    }
    isCapturing(): boolean {
        return this.capture !== null;
    }

    /** Native toPython(events): {"<name>/cpu_time": {name, value, average, stats}, ...}. */
    eventsToJson(): Record<string, { name: string; value: number; average: number; stats: Record<string, number> }> {
        const out: Record<string, { name: string; value: number; average: number; stats: Record<string, number> }> = {};
        for (const e of this.lastFrameEvents) {
            out[`${e.name}/cpu_time`] = { name: `${e.name}/cpu_time`, value: e.cpuTime, average: e.cpuTimeAverage, stats: statsToJson(e.computeCpuTimeStats()) };
            out[`${e.name}/gpu_time`] = { name: `${e.name}/gpu_time`, value: e.gpuTime, average: e.gpuTimeAverage, stats: statsToJson(e.computeGpuTimeStats()) };
        }
        return out;
    }

    /** Python surface with the native binding's names (`m.profiler`); `toPy` converts dict results. */
    pythonBindings(toPy: (v: unknown) => unknown = (v) => v): object {
        const self = this;
        return {
            get enabled() { return self.enabled; },
            set enabled(v: boolean) { self.enabled = !!v; },
            get paused() { return self.paused; },
            set paused(v: boolean) { self.paused = !!v; },
            get is_capturing() { return self.isCapturing(); },
            get events() { return toPy(self.eventsToJson()); },
            start_capture: (reservedFrames = 1000) => self.startCapture(reservedFrames),
            end_capture: () => { const c = self.endCapture(); return c ? toPy(c.toJson()) : null; },
            end_frame: () => self.endFrame(null),
            reset_stats: () => self.resetStats(),
        };
    }
}
