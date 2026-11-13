/**
 * Pass-through test of SimplePostFX: true when every effect is neutral, so execute() may blit src to dst.
 * vignetteAmount is an effect of the final pass (col *= max(0, 1 - r^2 * vignetteAmount)) and must be part of the test;
 * without it a vignette-only configuration was silently a no-op. starAmount is not listed: the star lobes are built from
 * the bloom pyramid, which only runs when bloomAmount > 0.
 */
export interface SimplePostFXEffectParams {
    bloomAmount: number;
    vignetteAmount: number;
    chromaticAberrationAmount: number;
    barrelDistortAmount: number;
    saturationCurve: readonly number[];
    colorOffset: readonly number[];
    colorScale: readonly number[];
    colorPower: readonly number[];
    colorOffsetScalar: number;
    colorScaleScalar: number;
    colorPowerScalar: number;
}

export function simplePostFXIsDefault(p: SimplePostFXEffectParams): boolean {
    return (
        p.bloomAmount === 0 &&
        p.vignetteAmount === 0 &&
        p.chromaticAberrationAmount === 0 &&
        p.barrelDistortAmount === 0 &&
        p.saturationCurve.every((v) => v === 1) &&
        p.colorOffset.every((v) => v === 0.5) &&
        p.colorScale.every((v) => v === 0.5) &&
        p.colorPower.every((v) => v === 0.5) &&
        p.colorOffsetScalar === 0 &&
        p.colorScaleScalar === 0 &&
        p.colorPowerScalar === 0
    );
}
