/**
 * Color-temperature / white-balance math mirroring Utils/Color/ColorUtils.h
 * (CAT02 von Kries adaptation in RGB Rec.709; D65 preserved at 6500 K).
 */

type Mat3 = [number, number, number, number, number, number, number, number, number]; // row-major

const kRGBtoXYZ_Rec709: Mat3 = [0.4123907992659595, 0.357584339383878, 0.1804807884018343, 0.2126390058715104, 0.7151686787677559, 0.0721923153607337, 0.0193308187155918, 0.1191947797946259, 0.9505321522496608];
const kXYZtoRGB_Rec709: Mat3 = [3.2409699419045213, -1.5373831775700935, -0.4986107602930033, -0.9692436362808798, 1.8759675015077206, 0.0415550574071756, 0.0556300796969936, -0.2039769588889765, 1.0569715142428784];
const kXYZtoLMS_CAT02: Mat3 = [0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.003, 0.0136, 0.9834];
const kLMStoXYZ_CAT02: Mat3 = [1.096123820835514, -0.278869000218287, 0.182745179382773, 0.454369041975359, 0.473533154307412, 0.072097803717229, -0.009627608738429, -0.005698031216113, 1.015325639954543];

export function mulMat3(a: Mat3, b: Mat3): Mat3 {
    const o = new Array<number>(9).fill(0) as Mat3;
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) o[r * 3 + c] = a[r * 3]! * b[c]! + a[r * 3 + 1]! * b[3 + c]! + a[r * 3 + 2]! * b[6 + c]!;
    return o;
}
export function mulMat3Vec(m: Mat3, v: [number, number, number]): [number, number, number] {
    return [m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2], m[6] * v[0] + m[7] * v[1] + m[8] * v[2]];
}
export function invertMat3(m: Mat3): Mat3 {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}

/** Mirrors colorTemperatureToXYZ (Kang et al. 2002 rational fits), T in [1667, 25000] K. */
export function colorTemperatureToXYZ(T: number, Y = 1): [number, number, number] {
    const t = T, t2 = t * t, t3 = t * t * t;
    const x = T < 4000 ? -0.2661239e9 / t3 - 0.234358e6 / t2 + 0.8776956e3 / t + 0.17991 : -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.24039;
    const x2 = x * x, x3 = x * x * x;
    const y = T < 2222 ? -1.1063814 * x3 - 1.3481102 * x2 + 2.18555832 * x - 0.20219683
        : T < 4000 ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
        : 3.081758 * x3 - 5.8733867 * x2 + 3.75112997 * x - 0.37001483;
    const xc = Math.fround(x), yc = Math.fround(y); // native casts the doubles to float before xyYtoXYZ
    return [xc * Y / yc, Y, (1 - xc - yc) * Y / yc];
}

/** Mirrors calculateWhiteBalanceTransformRGB_Rec709: 3x3 row-major RGB transform for target temperature T. */
export function calculateWhiteBalanceTransformRGB_Rec709(T: number): Mat3 {
    const MA = mulMat3(kXYZtoLMS_CAT02, kRGBtoXYZ_Rec709);
    const invMA = mulMat3(kXYZtoRGB_Rec709, kLMStoXYZ_CAT02);
    const wd = mulMat3Vec(kXYZtoLMS_CAT02, colorTemperatureToXYZ(6500));
    const ws = mulMat3Vec(kXYZtoLMS_CAT02, colorTemperatureToXYZ(T));
    const D: Mat3 = [wd[0] / ws[0], 0, 0, 0, wd[1] / ws[1], 0, 0, 0, wd[2] / ws[2]];
    return mulMat3(mulMat3(invMA, D), MA);
}
export type { Mat3 };
