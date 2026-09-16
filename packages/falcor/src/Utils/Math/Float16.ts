/**
 * Half-float conversion mirroring Utils/Math/Float16.cpp (OpenEXR algorithm:
 * round to nearest, ties up; subnormals preserved; NaN payload kept).
 * Every host-side fp16 pack in Falcor goes through this, so bit parity with
 * native material/vertex/light data depends on the exact rounding rule.
 */

const kF32 = new Float32Array(1);
const kU32 = new Uint32Array(kF32.buffer);

/** Mirrors float32ToFloat16 (bit pattern of the half). */
export function float32ToFloat16(value: number): number {
    kF32[0] = value;
    const i = kU32[0]! | 0;
    const s = (i >> 16) & 0x8000;
    let e = ((i >> 23) & 0xff) - (127 - 15);
    let m = i & 0x007fffff;

    if (e <= 0) {
        if (e < -10) return s; // below half subnormal range -> signed zero
        m = (m | 0x00800000) >> (1 - e);
        if (m & 0x00001000) m += 0x00002000; // round to nearest, "0.5" up (may renormalize)
        return s | (m >> 13);
    }
    if (e === 0xff - (127 - 15)) {
        if (m === 0) return s | 0x7c00; // inf
        m >>= 13;
        return s | 0x7c00 | m | (m === 0 ? 1 : 0); // NaN keeps payload, never collapses to inf
    }
    if (m & 0x00001000) {
        m += 0x00002000;
        if (m & 0x00800000) {
            m = 0; // significand overflow -> bump exponent
            e += 1;
        }
    }
    if (e > 30) return s | 0x7c00; // exponent overflow -> inf
    return s | (e << 10) | (m >> 13);
}

/** Mirrors float16ToFloat32 (half bit pattern -> number). */
export function float16ToFloat32(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exp = (bits >>> 10) & 0x1f;
    const mant = bits & 0x3ff;
    if (exp === 0) return sign * mant * 2 ** -24;
    if (exp === 31) return mant ? NaN : sign * Infinity;
    return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}
