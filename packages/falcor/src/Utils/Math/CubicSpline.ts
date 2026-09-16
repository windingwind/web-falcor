/**
 * Natural cubic spline mirroring Utils/Math/CubicSpline.h (uniform-parameter
 * setup + Horner interpolate) over N-component lanes stored flat.
 */

/** Natural cubic spline over N-component lanes (CubicSpline.h setup/interpolate). */
export class CubicSpline {
    private a: Float32Array;
    private b: Float32Array;
    private c: Float32Array;
    private d: Float32Array;
    private readonly lanes: number;

    constructor(controlPoints: ArrayLike<number>, pointCount: number, lanes: number) {
        this.lanes = lanes;
        const n = pointCount;
        this.a = new Float32Array(n * lanes);
        this.b = new Float32Array(n * lanes);
        this.c = new Float32Array(n * lanes);
        this.d = new Float32Array(n * lanes);
        const gamma = new Float32Array(n);
        const delta = new Float32Array(n * lanes);
        const D = new Float32Array(n * lanes);

        gamma[0] = 0.5;
        for (let i = 1; i < n - 1; i++) gamma[i] = 1 / (4 - gamma[i - 1]!);
        gamma[n - 1] = 1 / (2 - gamma[n - 2]!);

        for (let l = 0; l < lanes; l++) {
            delta[l] = 3 * (controlPoints[lanes + l]! - controlPoints[l]!) * gamma[0]!;
        }
        for (let i = 1; i < n; i++) {
            const index = i === n - 1 ? i : i + 1;
            for (let l = 0; l < lanes; l++) {
                delta[i * lanes + l] = (3 * (controlPoints[index * lanes + l]! - controlPoints[(i - 1) * lanes + l]!) - delta[(i - 1) * lanes + l]!) * gamma[i]!;
            }
        }
        for (let l = 0; l < lanes; l++) D[(n - 1) * lanes + l] = delta[(n - 1) * lanes + l]!;
        for (let i = n - 2; i >= 0; i--) {
            for (let l = 0; l < lanes; l++) {
                D[i * lanes + l] = delta[i * lanes + l]! - gamma[i]! * D[(i + 1) * lanes + l]!;
            }
        }
        for (let i = 0; i < n - 1; i++) {
            for (let l = 0; l < lanes; l++) {
                const p0 = controlPoints[i * lanes + l]!;
                const p1 = controlPoints[(i + 1) * lanes + l]!;
                this.a[i * lanes + l] = p0;
                this.b[i * lanes + l] = D[i * lanes + l]!;
                this.c[i * lanes + l] = 3 * (p1 - p0) - 2 * D[i * lanes + l]! - D[(i + 1) * lanes + l]!;
                this.d[i * lanes + l] = 2 * (p0 - p1) + D[i * lanes + l]! + D[(i + 1) * lanes + l]!;
            }
        }
    }

    /** Polynomial coefficients of one section/lane (a + b t + c t^2 + d t^3). */
    coefficients(section: number, lane: number): { a: number; b: number; c: number; d: number } {
        const i = section * this.lanes + lane;
        return { a: this.a[i]!, b: this.b[i]!, c: this.c[i]!, d: this.d[i]! };
    }

    /** Horner evaluation within a section at t in [0,1] (one lane). */
    interpolate(section: number, t: number, lane: number): number {
        const i = section * this.lanes + lane;
        return ((this.d[i]! * t + this.c[i]!) * t + this.b[i]!) * t + this.a[i]!;
    }
}
