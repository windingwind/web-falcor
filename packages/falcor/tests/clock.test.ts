/**
 * Clock unit tests: FPS-simulation stepping on the native tick grid (the
 * m.clock.framerate/pause/frame oracle recipe), deferred sets, realtime scale.
 */

import { describe, expect, it } from "vitest";
import { Clock } from "../src/Utils/Timing/Clock.js";

describe("Clock", () => {
    it("fps-simulation: frame stepping lands on exact frame times", () => {
        const c = new Clock(() => 0);
        c.setFramerate(10);
        c.pause();
        c.setFrame(30);
        expect(c.getTime()).toBeCloseTo(3.0, 12);
        c.step();
        expect(c.getFrame()).toBe(31);
        expect(c.getTime()).toBeCloseTo(3.1, 12);
        c.step(-2);
        expect(c.getFrame()).toBe(29);
        expect(c.getTime()).toBeCloseTo(2.9, 12);
    });

    it("setTime quantizes to the frame grid when simulating fps", () => {
        const c = new Clock(() => 0);
        c.setFramerate(10);
        c.setTime(3.14);
        expect(c.getFrame()).toBe(31);
        expect(c.getTime()).toBeCloseTo(3.1, 12);
    });

    it("deferred frame set applies on the next tick", () => {
        const c = new Clock(() => 0);
        c.setFramerate(10);
        c.pause();
        c.setFrame(50, true);
        expect(c.getFrame()).toBe(0);
        c.tick();
        expect(c.getFrame()).toBe(50);
        expect(c.getTime()).toBeCloseTo(5.0, 12);
        c.tick(); // paused: no further advance
        expect(c.getFrame()).toBe(50);
    });

    it("realtime mode advances by timer delta times scale", () => {
        let calls = 0;
        const c = new Clock(() => calls++ * 0.016);
        c.setTimeScale(2);
        c.tick(); // step: one timer call -> dt 0.016 * scale 2
        expect(c.getTime()).toBeCloseTo(0.032, 9);
        expect(c.getDelta()).toBeCloseTo(0.032, 9);
    });

    it("stop rewinds and pauses; paused tick holds", () => {
        const c = new Clock(() => 0);
        c.setFramerate(10);
        c.setFrame(20);
        c.stop();
        expect(c.getTime()).toBe(0);
        expect(c.isPaused()).toBe(true);
        c.tick();
        expect(c.getTime()).toBe(0);
    });

    it("python property surface maps to the setters", () => {
        const c = new Clock(() => 0);
        c.framerate = 10;
        c.frame = 30;
        expect(c.time).toBeCloseTo(3.0, 12);
        c.time = 5.0;
        expect(c.frame).toBe(50);
    });
});
