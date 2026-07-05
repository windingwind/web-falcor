/**
 * GPU unit-test registry, mirroring FalcorTest's GPU_TEST pattern.
 * Test files call gpuTest("name", fn); the harness runs them sequentially
 * against a shared Device.
 */

import type { Device } from "@web-falcor/falcor";

export interface GpuTestContext {
    device: Device;
}

export type GpuTestFn = (ctx: GpuTestContext) => Promise<void> | void;

export interface GpuTestCase {
    name: string;
    fn: GpuTestFn;
    skip?: string;
}

export const tests: GpuTestCase[] = [];

export function gpuTest(name: string, fn: GpuTestFn): void {
    tests.push({ name, fn });
}

/** Register a test that is expected to be skipped (feature unavailable etc.). */
gpuTest.skipIf = (condition: (ctx: GpuTestContext) => string | null, name: string, fn: GpuTestFn): void => {
    tests.push({
        name,
        fn: async (ctx) => {
            const reason = condition(ctx);
            if (reason) throw new SkipError(reason);
            await fn(ctx);
        },
    });
};

export class SkipError extends Error {}

// ---- assertion helpers ----

function stringify(value: unknown): string {
    return JSON.stringify(value, (_k, v) => (typeof v === "bigint" ? `${v}n` : v));
}

export function expectEq(actual: unknown, expected: unknown, what = "value"): void {
    const a = stringify(actual);
    const e = stringify(expected);
    if (a !== e) throw new Error(`expected ${what} == ${e}, got ${a}`);
}

export function expectClose(actual: number, expected: number, eps = 1e-5, what = "value"): void {
    if (!(Math.abs(actual - expected) <= eps)) {
        throw new Error(`expected ${what} ≈ ${expected} (eps ${eps}), got ${actual}`);
    }
}

export function expectArrayEq(actual: ArrayLike<number>, expected: ArrayLike<number>, what = "array"): void {
    if (actual.length !== expected.length) {
        throw new Error(`expected ${what}.length == ${expected.length}, got ${actual.length}`);
    }
    for (let i = 0; i < expected.length; i++) {
        if (actual[i] !== expected[i]) {
            throw new Error(`${what}[${i}]: expected ${expected[i]}, got ${actual[i]}`);
        }
    }
}

export function expectArrayClose(actual: ArrayLike<number>, expected: ArrayLike<number>, eps = 1e-5, what = "array"): void {
    if (actual.length !== expected.length) {
        throw new Error(`expected ${what}.length == ${expected.length}, got ${actual.length}`);
    }
    for (let i = 0; i < expected.length; i++) {
        if (!(Math.abs((actual[i] as number) - (expected[i] as number)) <= eps)) {
            throw new Error(`${what}[${i}]: expected ≈${expected[i]} (eps ${eps}), got ${actual[i]}`);
        }
    }
}
