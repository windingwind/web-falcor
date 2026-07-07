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

/** PNG artifacts a test asked to save; the Node runner writes them to disk. */
export interface Artifact {
    name: string;
    b64: string;
    width: number;
    height: number;
}
export const artifacts: Artifact[] = [];

/**
 * Saves a texture readback as a viewable PNG (written to tests/gpu/out/<name>.png
 * by the Node runner). `pixels` is the raw readback; pass bgra=true for a
 * BGRA8 present target (channels are swapped to RGBA for the PNG).
 */
export async function saveArtifact(
    name: string,
    pixels: ArrayLike<number>,
    width: number,
    height: number,
    bgra = true,
): Promise<void> {
    const rgba = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < width * height; i++) {
        const r = bgra ? pixels[i * 4 + 2]! : pixels[i * 4]!;
        const b = bgra ? pixels[i * 4]! : pixels[i * 4 + 2]!;
        rgba[i * 4] = r;
        rgba[i * 4 + 1] = pixels[i * 4 + 1]!;
        rgba[i * 4 + 2] = b;
        rgba[i * 4 + 3] = 255; // opaque (present targets carry no meaningful alpha)
    }
    const canvas = new OffscreenCanvas(width, height);
    canvas.getContext("2d")!.putImageData(new ImageData(rgba, width, height), 0, 0);
    const blob = await canvas.convertToBlob({ type: "image/png" });
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let bin = "";
    for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!);
    artifacts.push({ name, b64: btoa(bin), width, height });
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
