/**
 * PhysicalSky: analytic clear-sky model of Preetham, Shirley and Smits (1999, "A Practical Analytic Model for
 * Daylight") with a sun disk, as a background pass and as an environment map.
 * Native Falcor 8 has no sky model (Falcor 4's SkyBox pass was removed upstream); this is a web-falcor addition.
 *
 * Pass   : inputs  color (optional), posW (optional; w = 0 marks background)
 *          outputs dst (RGBA32Float): the sky radiance seen through each pixel; where posW marks geometry the
 *                  color input is passed through (without posW the sky covers the whole frame).
 * Env map: createPhysicalSkyEnvMap(device, params, width) evaluates the same model on the CPU into a lat-long
 *          EnvMap (Falcor's world_to_latlong_map convention), usable with scene.setEnvMap() for lighting, the
 *          path tracers' background and EnvMapSampler importance sampling.
 * Units: Preetham's luminance is in kcd/m^2; outputs are luminance * skyScale (linear sRGB, D65).
 * Up axis is +y (glTF / Falcor default). Below the horizon the horizon radiance times groundAlbedo is returned.
 * Sun disk: angular radius sunAngularRadiusDeg, radiance sunIntensity * T(airmass) with an RGB Rayleigh +
 * turbidity-scaled aerosol transmittance (Kasten-Young airmass); an approximation, not part of Preetham's model.
 */

import {
    ComputePass,
    EnvMap,
    FieldFlags,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/PhysicalSky/PhysicalSky.cs.slang";

export interface PhysicalSkyParams {
    /** Direction towards the sun (normalised internally); y is up. */
    sunDir: [number, number, number];
    /** Atmospheric turbidity, valid range ~[1.7, 10]. */
    turbidity: number;
    skyScale: number;
    sunIntensity: number;
    sunAngularRadiusDeg: number;
    groundAlbedo: [number, number, number];
}

export const kDefaultPhysicalSky: PhysicalSkyParams = {
    sunDir: [0.3, 0.6, 0.4],
    turbidity: 2.5,
    skyScale: 0.05,
    sunIntensity: 2000,
    sunAngularRadiusDeg: 0.27,
    groundAlbedo: [0.3, 0.3, 0.3],
};

const norm3 = (v: readonly number[]): [number, number, number] => {
    const l = Math.hypot(v[0]!, v[1]!, v[2]!) || 1;
    return [v[0]! / l, v[1]! / l, v[2]! / l];
};

/** Preetham's Perez coefficients (A..E) for Y, x, y at turbidity T, and the zenith values (Y in kcd/m^2). */
export function preethamCoefficients(T: number, thetaS: number): { Y: number[]; x: number[]; y: number[]; zenith: [number, number, number] } {
    const Y = [0.1787 * T - 1.463, -0.3554 * T + 0.4275, -0.0227 * T + 5.3251, 0.1206 * T - 2.5771, -0.067 * T + 0.3703];
    const x = [-0.0193 * T - 0.2592, -0.0665 * T + 0.0008, -0.0004 * T + 0.2125, -0.0641 * T - 0.8989, -0.0033 * T + 0.0452];
    const y = [-0.0167 * T - 0.2608, -0.095 * T + 0.0092, -0.0079 * T + 0.2102, -0.0441 * T - 1.6537, -0.0109 * T + 0.0529];
    const chi = (4 / 9 - T / 120) * (Math.PI - 2 * thetaS);
    const Yz = (4.0453 * T - 4.971) * Math.tan(chi) - 0.2155 * T + 2.4192;
    const t = thetaS, t2 = t * t, t3 = t2 * t, T2 = T * T;
    const xz = T2 * (0.00166 * t3 - 0.00375 * t2 + 0.00209 * t) + T * (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * t + 0.00394)
        + (0.11693 * t3 - 0.21196 * t2 + 0.06052 * t + 0.25886);
    const yz = T2 * (0.00275 * t3 - 0.0061 * t2 + 0.00317 * t) + T * (-0.04214 * t3 + 0.0897 * t2 - 0.04153 * t + 0.00516)
        + (0.15346 * t3 - 0.26756 * t2 + 0.0667 * t + 0.26688);
    return { Y, x, y, zenith: [Yz, xz, yz] };
}

const perez = (c: number[], cosTheta: number, gamma: number, cosGamma: number) =>
    (1 + c[0]! * Math.exp(c[1]! / Math.max(cosTheta, 1e-3))) * (1 + c[2]! * Math.exp(c[3]! * gamma) + c[4]! * cosGamma * cosGamma);

function xyYToLinearSrgb(x: number, y: number, Yv: number): [number, number, number] {
    const X = (x * Yv) / y, Z = ((1 - x - y) * Yv) / y;
    return [3.2404542 * X - 1.5371385 * Yv - 0.4985314 * Z, -0.969266 * X + 1.8760108 * Yv + 0.041556 * Z, 0.0556434 * X - 0.2040259 * Yv + 1.0572252 * Z];
}

/** Sun transmittance (RGB) along the view to the sun: Rayleigh + aerosol optical depth, Kasten-Young airmass. */
export function sunTransmittance(T: number, cosThetaS: number): [number, number, number] {
    const thetaDeg = (Math.acos(Math.min(Math.max(cosThetaS, 0), 1)) * 180) / Math.PI;
    const m = 1 / (Math.max(cosThetaS, 0) + 0.50572 * Math.pow(Math.max(96.07995 - thetaDeg, 1e-3), -1.6364));
    const tauR = [0.041, 0.0973, 0.235]; // Rayleigh optical depth at ~680/550/440 nm
    const tauA = 0.08 * Math.max(T - 1, 0); // aerosol optical depth grows with turbidity (approximation)
    return [0, 1, 2].map((i) => Math.exp(-m * (tauR[i]! + tauA))) as [number, number, number];
}

/** CPU reference of the shader (same formulas): radiance towards direction `dir` (y up). */
export function evalPhysicalSky(dir: readonly number[], params: PhysicalSkyParams = kDefaultPhysicalSky, withSun = true): [number, number, number] {
    const s = norm3(params.sunDir);
    const d = norm3(dir);
    const below = d[1] < 0;
    const dy = Math.max(d[1], 0);
    const dd = norm3([d[0], dy, d[2]]);
    const thetaS = Math.acos(Math.min(Math.max(s[1], 0), 1));
    const c = preethamCoefficients(params.turbidity, thetaS);
    const cosTheta = dd[1];
    const cosGamma = Math.min(Math.max(dd[0] * s[0] + dd[1] * s[1] + dd[2] * s[2], -1), 1);
    const gamma = Math.acos(cosGamma);
    const cosTS = Math.cos(thetaS);
    const f = (k: number[]) => perez(k, cosTheta, gamma, cosGamma) / perez(k, 1, thetaS, cosTS);
    const Yv = c.zenith[0] * f(c.Y), x = c.zenith[1] * f(c.x), y = c.zenith[2] * f(c.y);
    let rgb = xyYToLinearSrgb(x, y, Yv).map((v) => Math.max(v, 0) * params.skyScale) as [number, number, number];
    if (below) {
        rgb = rgb.map((v, i) => v * params.groundAlbedo[i]!) as [number, number, number];
    } else if (withSun) {
        const cosG = d[0] * s[0] + d[1] * s[1] + d[2] * s[2];
        if (cosG >= Math.cos((params.sunAngularRadiusDeg * Math.PI) / 180) && s[1] > 0) {
            const tr = sunTransmittance(params.turbidity, s[1]);
            rgb = rgb.map((v, i) => v + params.sunIntensity * tr[i]!) as [number, number, number];
        }
    }
    return rgb;
}

/** Lat-long EnvMap of the sky (height = width / 2). The sun disk is included when it covers at least one texel
 *  (otherwise use withSun=false and an analytic DirectionalLight for the sun). */
export function createPhysicalSkyEnvMap(device: Device, params: PhysicalSkyParams = kDefaultPhysicalSky, width = 1024, withSun = false): EnvMap {
    const height = Math.max(1, width >> 1);
    const data = new Float32Array(width * height * 4);
    for (let j = 0; j < height; j++) {
        const theta = (Math.PI * (j + 0.5)) / height;
        for (let i = 0; i < width; i++) {
            const phi = Math.PI * (2 * ((i + 0.5) / width) - 1);
            // latlong_map_to_world (Utils/Math/MathHelpers.slang)
            const dir = [Math.sin(theta) * Math.sin(phi), Math.cos(theta), -Math.sin(theta) * Math.cos(phi)];
            const c = evalPhysicalSky(dir, params, withSun);
            const o = (j * width + i) * 4;
            data[o] = c[0]; data[o + 1] = c[1]; data[o + 2] = c[2]; data[o + 3] = 1;
        }
    }
    return new EnvMap(device, { width, height, data } as ConstructorParameters<typeof EnvMap>[1]);
}

export class PhysicalSky extends RenderPass {
    private p: PhysicalSkyParams = { ...kDefaultPhysicalSky };
    private pass: ComputePass | null = null;
    private dummy: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(this.p) as (keyof PhysicalSkyParams)[]) (this.p as unknown as Record<string, unknown>)[k] = props.get(k, this.p[k]);
    }

    get params(): PhysicalSkyParams {
        return { ...this.p };
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p });
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Turbidity", this.p.turbidity, 1.7, 10, 0.01, (v) => (this.p.turbidity = v));
        ui.slider("Sky scale", this.p.skyScale, 0, 1, 0.001, (v) => (this.p.skyScale = v));
        ui.slider("Sun intensity", this.p.sunIntensity, 0, 1e5, 1, (v) => (this.p.sunIntensity = v));
        ui.slider("Sun radius (deg)", this.p.sunAngularRadiusDeg, 0, 5, 0.01, (v) => (this.p.sunAngularRadiusDeg = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("color", "Scene color (optional)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("posW", "World position, w = 0 marks background (optional)").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("dst", "Sky behind the scene")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst");
        if (!dst) return;
        const color = rd.getTexture("color");
        const posW = rd.getTexture("posW");
        const W = dst.width, H = dst.height;
        if (!this.pass) this.pass = ComputePass.create(this.device, { path: kShaderFile });
        // camera basis (defaults: looking down -z) for per-pixel view rays
        let fwd = [0, 0, -1], up = [0, 1, 0], tanY = Math.tan(Math.PI / 8), aspect = W / H;
        if (this.scene) {
            const cam = this.scene.camera;
            const p = cam.getPosition(), t = cam.getTarget(), u = cam.getUpVector();
            fwd = norm3([t.x - p.x, t.y - p.y, t.z - p.z]);
            up = [u.x, u.y, u.z];
            tanY = Math.tan(0.5 * cam.getFovY());
            aspect = cam.getAspectRatio();
        }
        const right = norm3([fwd[1]! * up[2]! - fwd[2]! * up[1]!, fwd[2]! * up[0]! - fwd[0]! * up[2]!, fwd[0]! * up[1]! - fwd[1]! * up[0]!]);
        const trueUp = [right[1] * fwd[2]! - right[2] * fwd[1]!, right[2] * fwd[0]! - right[0] * fwd[2]!, right[0] * fwd[1]! - right[1] * fwd[0]!];
        const s = norm3(this.p.sunDir);
        const thetaS = Math.acos(Math.min(Math.max(s[1], 0), 1));
        const c = preethamCoefficients(this.p.turbidity, thetaS);
        const tr = sunTransmittance(this.p.turbidity, s[1]);
        const cb = this.pass.getRootVar()["CB"]!;
        cb["gFrameDim"] = [W, H];
        cb["gHasColor"] = color ? 1 : 0;
        cb["gHasPos"] = posW ? 1 : 0;
        cb["gCamRight"] = right.map((v) => v * tanY * aspect);
        cb["gCamUp"] = trueUp.map((v) => v * tanY);
        cb["gCamFwd"] = fwd;
        cb["gSunDir"] = s;
        cb["gThetaS"] = thetaS;
        cb["gCoefY0"] = c.Y.slice(0, 4);
        cb["gCoefX0"] = c.x.slice(0, 4);
        cb["gCoefYy0"] = c.y.slice(0, 4);
        cb["gCoefE"] = [c.Y[4]!, c.x[4]!, c.y[4]!, 0];
        cb["gZenith"] = c.zenith;
        cb["gSkyScale"] = this.p.skyScale;
        cb["gSunRadiance"] = tr.map((v) => v * this.p.sunIntensity * (s[1] > 0 ? 1 : 0));
        cb["gCosSunRadius"] = Math.cos((this.p.sunAngularRadiusDeg * Math.PI) / 180);
        cb["gGround"] = this.p.groundAlbedo;
        const root = this.pass.getRootVar();
        if ((!color || !posW) && !this.dummy) {
            this.dummy = new Texture(this.device, { type: ResourceType.Texture2D, width: 1, height: 1, format: ResourceFormat.RGBA32Float,
                bindFlags: ResourceBindFlags.ShaderResource, name: "PhysicalSky::dummy" });
        }
        root["gColor"] = color ?? this.dummy!; // unconnected optional inputs: gHasColor/gHasPos say they are unused
        root["gPosW"] = posW ?? this.dummy!;
        root["gDst"] = dst;
        this.pass.execute(ctx, W, H);
    }
}

registerRenderPass("PhysicalSky", (device, props) => new PhysicalSky(device, props), { desc: "Preetham analytic sky + sun disk (background / env map)" });
