/**
 * Port of hypothesis::chi2_test (Wenzel Jakob's hypothesis.h, used by
 * FalcorTest): Pearson's chi^2 test with low-expectation cells pooled.
 */

/** Regularized lower incomplete gamma P(a, x) (Numerical Recipes gser/gcf). */
function gammaP(a: number, x: number): number {
    if (x <= 0) return 0;
    const lnGammaA = lnGamma(a);
    if (x < a + 1) {
        let sum = 1 / a;
        let del = sum;
        for (let n = 1; n < 1000; n++) {
            del *= x / (a + n);
            sum += del;
            if (Math.abs(del) < Math.abs(sum) * 1e-15) break;
        }
        return sum * Math.exp(-x + a * Math.log(x) - lnGammaA);
    }
    let b = x + 1 - a;
    let c = 1e300;
    let d = 1 / b;
    let h = d;
    for (let i = 1; i < 1000; i++) {
        const an = -i * (i - a);
        b += 2;
        d = an * d + b;
        if (Math.abs(d) < 1e-300) d = 1e-300;
        c = b + an / c;
        if (Math.abs(c) < 1e-300) c = 1e-300;
        d = 1 / d;
        const del = d * c;
        h *= del;
        if (Math.abs(del - 1) < 1e-15) break;
    }
    return 1 - Math.exp(-x + a * Math.log(x) - lnGammaA) * h;
}

/** Lanczos log-gamma. */
function lnGamma(x: number): number {
    const g = [676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lnGamma(1 - x);
    x -= 1;
    let a = 0.99999999999980993;
    const t = x + 7.5;
    for (let i = 0; i < 8; i++) a += g[i]! / (x + i + 1);
    return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

export function chi2Test(
    cellCount: number,
    obs: ArrayLike<number>,
    exp: ArrayLike<number>,
    sampleCount: number,
    minExpFrequency: number,
    significanceLevel: number,
    numTests = 1,
): { success: boolean; report: string } {
    const cells = Array.from({ length: cellCount }, (_, i) => i).sort((a, b) => exp[a]! - exp[b]!);
    let pooledObs = 0;
    let pooledExp = 0;
    let chsq = 0;
    let dof = 0;
    for (const i of cells) {
        if (exp[i] === 0) {
            if (obs[i]! > sampleCount * 1e-5) return { success: false, report: `nonzero observed frequency ${obs[i]} in cell ${i} with zero expected frequency` };
        } else if (exp[i]! < minExpFrequency || (pooledExp > 0 && pooledExp < minExpFrequency)) {
            pooledObs += obs[i]!;
            pooledExp += exp[i]!;
        } else {
            const diff = obs[i]! - exp[i]!;
            chsq += (diff * diff) / exp[i]!;
            dof++;
        }
    }
    if (pooledExp > 0 || pooledObs > 0) {
        const diff = pooledObs - pooledExp;
        chsq += (diff * diff) / pooledExp;
        dof++;
    }
    dof--;
    if (dof <= 0) return { success: false, report: `number of degrees of freedom (${dof}) is too small` };
    const pval = 1 - gammaP(dof / 2, chsq / 2);
    const alpha = 1 - Math.pow(1 - significanceLevel, 1 / numTests);
    const report = `chi^2 = ${chsq.toFixed(3)}, dof = ${dof}, p-value = ${pval.toExponential(3)}, alpha = ${alpha}`;
    return { success: Number.isFinite(pval) && pval >= alpha, report };
}
