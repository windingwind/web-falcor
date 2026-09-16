/** Mirrors Utils/Timing/TimeReport.h: named phase durations (seconds) logged as a report. */

import { Logger } from "../Logger.js";

export class TimeReport {
    private lastMeasureTime = 0;
    private measurements: [string, number][] = [];
    private total = 0;

    constructor(private readonly now: () => number = () => performance.now()) {
        this.reset();
    }

    reset(): void {
        this.lastMeasureTime = this.now();
        this.measurements = [];
        this.total = 0;
    }

    resetTimer(): void {
        this.lastMeasureTime = this.now();
        this.total = 0;
    }

    /** Records the time since the last measure() (or reset) under `name`. */
    measure(name: string): void {
        const t = this.now();
        this.measurements.push([name, (t - this.lastMeasureTime) / 1000]);
        this.lastMeasureTime = t;
    }

    /** Appends the sum of all measurements as a "Total" row (native ignores the name argument). */
    addTotal(_name = "Total"): void {
        this.total = this.measurements.reduce((t, [, d]) => t + d, 0);
        this.measurements.push(["Total", this.total]);
    }

    getMeasurements(): readonly [string, number][] {
        return this.measurements;
    }

    printToLog(): void {
        for (const [task, duration] of this.measurements) {
            const pct = this.total > 0 ? `, ${(100 * duration / this.total).toFixed(6)}% of total` : "";
            Logger.info(`${(task + ":").padEnd(25)} ${duration.toFixed(6)} s${pct}`);
        }
    }
}
