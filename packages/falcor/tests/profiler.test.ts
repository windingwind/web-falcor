import { describe, expect, it } from "vitest";
import { Profiler, computeStats } from "../src/Core/API/Profiler.js";
import { TimeReport } from "../src/Utils/Timing/TimeReport.js";
import { FrameRate } from "../src/Utils/Timing/FrameRate.js";

/** CPU-only profiler (no device) driven by a fake clock in ms. */
function makeProfiler(): { p: Profiler; clock: { t: number } } {
    const clock = { t: 0 };
    return { p: new Profiler(null, () => clock.t), clock };
}

describe("Profiler events (native Profiler.cpp semantics)", () => {
    it("nests names with '/', accumulates CPU time and orders events by first start", () => {
        const { p, clock } = makeProfiler();
        p.startEvent("frame");
        clock.t = 1;
        p.startEvent("passA");
        clock.t = 3;
        p.endEvent("passA");
        p.startEvent("passB");
        clock.t = 4;
        p.endEvent("passB");
        clock.t = 5;
        p.endEvent("frame");
        p.endFrame(null);
        const names = p.getEvents().map((e) => e.name);
        expect(names).toEqual(["/frame", "/frame/passA", "/frame/passB"]);
        expect(p.getEvents().map((e) => e.cpuTime)).toEqual([5, 2, 1]);
        expect(p.getEvents().map((e) => e.level)).toEqual([0, 1, 1]);
        expect(p.getEvents()[1]!.shortName).toBe("passA");
        // Legacy per-pass view: children of the top-level graph event.
        expect([...p.getStats().keys()]).toEqual(["passA", "passB"]);
    });

    it("sums repeated events within a frame and rejects '/' in names", () => {
        const { p, clock } = makeProfiler();
        p.startEvent("x");
        clock.t = 2;
        p.endEvent("x");
        p.startEvent("x");
        clock.t = 5;
        p.endEvent("x");
        p.startEvent("bad/name");
        p.endEvent("bad/name");
        p.endFrame(null);
        expect(p.getEvents().map((e) => e.name)).toEqual(["/x"]);
        expect(p.getEvents()[0]!.cpuTime).toBe(5);
    });

    it("EMA uses sigma 0.98 after the first measurement; history feeds stats", () => {
        const { p, clock } = makeProfiler();
        const run = (ms: number) => {
            p.startEvent("e");
            clock.t += ms;
            p.endEvent("e");
            p.endFrame(null);
        };
        run(4);
        expect(p.getEvent("/e").cpuTimeAverage).toBe(4);
        run(2);
        expect(p.getEvent("/e").cpuTimeAverage).toBeCloseTo(0.98 * 4 + 0.02 * 2, 6);
        run(6);
        const s = p.getEvent("/e").computeCpuTimeStats();
        expect([s.min, s.max, s.mean]).toEqual([2, 6, 4]);
        expect(s.stdDev).toBeCloseTo(Math.sqrt(8 / 3), 5);
        expect(p.getEvent("/e").gpuTime).toBe(0);
        p.resetStats();
        run(1);
        expect(p.getEvent("/e").computeCpuTimeStats().max).toBe(0); // history cleared at that frame
    });

    it("paused profiler records nothing; disabled profiler ignores events", () => {
        const { p, clock } = makeProfiler();
        p.setPaused(true);
        p.startEvent("e");
        clock.t = 3;
        p.endEvent("e");
        p.endFrame(null);
        expect(p.getEvent("/e").cpuTime).toBe(0);
        p.setPaused(false);
        p.setEnabled(false);
        p.startEvent("f");
        p.endEvent("f");
        p.endFrame(null);
        expect(p.findEvent("/f")).toBeUndefined();
    });

    it("captures lanes over frames (first captured frame only fixes the event set)", () => {
        const { p, clock } = makeProfiler();
        const run = (ms: number) => {
            p.startEvent("e");
            clock.t += ms;
            p.endEvent("e");
            p.endFrame(null);
        };
        run(1); // populate last-frame events
        p.startCapture();
        expect(p.isCapturing()).toBe(true);
        run(2);
        run(3);
        run(4);
        const c = p.endCapture()!;
        expect(p.isCapturing()).toBe(false);
        expect(c.frameCount).toBe(2);
        expect(c.lanes.map((l) => l.name)).toEqual(["/e/cpu_time", "/e/gpu_time"]);
        expect(c.lanes[0]!.records).toEqual([3, 4]);
        expect(c.lanes[0]!.stats.mean).toBe(3.5);
        const json = JSON.parse(c.toJsonString());
        expect(json.frame_count).toBe(2);
        expect(json.events["/e/cpu_time"].stats.std_dev).toBeCloseTo(0.5, 6);
        expect(p.endCapture()).toBeNull();
    });

    it("exposes the python binding names", () => {
        const { p, clock } = makeProfiler();
        p.startEvent("e");
        clock.t = 2;
        p.endEvent("e");
        p.endFrame(null);
        const py = p.pythonBindings() as { paused: boolean; enabled: boolean; is_capturing: boolean; events: Record<string, { value: number }>; start_capture: () => void; end_capture: () => unknown };
        py.paused = true;
        expect(p.isPaused()).toBe(true);
        py.paused = false;
        expect(py.events["/e/cpu_time"]!.value).toBe(2);
        py.start_capture();
        expect(py.is_capturing).toBe(true);
        expect(py.end_capture()).toMatchObject({ frame_count: 0 });
    });

    it("computeStats handles empty and single inputs", () => {
        expect(computeStats([])).toEqual({ min: 0, max: 0, mean: 0, stdDev: 0 });
        expect(computeStats([7])).toEqual({ min: 7, max: 7, mean: 7, stdDev: 0 });
    });
});

describe("TimeReport / FrameRate", () => {
    it("TimeReport measures phases in seconds and appends a total", () => {
        let t = 0;
        const r = new TimeReport(() => t);
        t = 1500;
        r.measure("load");
        t = 2000;
        r.measure("build");
        r.addTotal();
        expect(r.getMeasurements()).toEqual([["load", 1.5], ["build", 0.5], ["Total", 2]]);
    });
    it("FrameRate averages the frame window (native slot indexing: frame N writes slot N % 60)", () => {
        let t = 0;
        const fr = new FrameRate(() => t);
        for (let i = 0; i < 10; i++) {
            t += 20;
            fr.newFrame();
        }
        expect(fr.getFrameCount()).toBe(10);
        expect(fr.getLastFrameTime()).toBeCloseTo(0.02, 9);
        // Slots 0..9 are averaged but frame 10 landed in slot 10: 9 samples over 10 (native quirk).
        expect(fr.getAverageFrameTime()).toBeCloseTo(0.018, 9);
        for (let i = 0; i < 51; i++) {
            t += 20;
            fr.newFrame();
        }
        expect(fr.getAverageFrameTime()).toBeCloseTo(0.02, 9); // full window
        expect(fr.getMsg()).toBe("50.0 FPS (20.0 ms/frame)");
        expect(fr.getMsg(true).endsWith(", VSync")).toBe(true);
    });
});
