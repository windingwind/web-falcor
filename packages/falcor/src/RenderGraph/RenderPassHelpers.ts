/**
 * Render pass IO sizing mirroring Falcor/RenderGraph/RenderPassHelpers.h.
 */

export enum IOSize {
    Default = 0, // reflected size 0 -> the graph's output dimensions
    Fixed = 1,
    Full = 2,
    Half = 3,
    Quarter = 4,
    Double = 5,
}

/** Parses the python-side property value ('Half', 'Fixed', ...) or a raw enum value. */
export function parseIOSize(value: string | number | undefined, fallback = IOSize.Default): IOSize {
    if (value === undefined) return fallback;
    if (typeof value === "number") return value;
    return IOSize[value as keyof typeof IOSize] ?? fallback;
}

/** Mirrors RenderPassHelpers::calculateIOSize (0,0 = use graph default). */
export function calculateIOSize(selection: IOSize, fixedSize: [number, number], windowSize: [number, number]): [number, number] {
    if (selection === IOSize.Fixed) return [fixedSize[0], fixedSize[1]];
    if (selection === IOSize.Full) return [windowSize[0], windowSize[1]];
    if (selection === IOSize.Half) return [windowSize[0] >>> 1, windowSize[1] >>> 1];
    if (selection === IOSize.Quarter) return [windowSize[0] >>> 2, windowSize[1] >>> 2];
    if (selection === IOSize.Double) return [windowSize[0] * 2, windowSize[1] * 2];
    return [0, 0];
}
