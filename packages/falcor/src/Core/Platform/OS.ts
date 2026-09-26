/**
 * Mirrors the bit helpers of Core/Platform/OS.h (the rest of OS.h is file and process
 * access the browser sandbox doesn't have; see docs §9).
 */

/** Index of the most significant set bit, or 0 if no bits are set. */
export function bitScanReverse(a: number): number {
    return a >>> 0 === 0 ? 0 : 31 - Math.clz32(a);
}

/** Index of the least significant set bit, or 0 if no bits are set. */
export function bitScanForward(a: number): number {
    const v = a >>> 0;
    return v === 0 ? 0 : 31 - Math.clz32(v & -v);
}

/** Number of set bits. */
export function popcount(a: number): number {
    let v = a >>> 0;
    v -= (v >>> 1) & 0x55555555;
    v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
    return (Math.imul((v + (v >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24) >>> 0;
}
