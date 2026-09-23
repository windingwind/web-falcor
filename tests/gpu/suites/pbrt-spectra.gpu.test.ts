/**
 * Spectral parameters and environment lights in the pbrt-v4 importer, which
 * native converts through Utils/Color/Spectrum (CIE 1931 + Rec.709) and the
 * PBRTImporter's EnvMapConverter (equal-area octahedral -> lat-long).
 *
 * Both fixtures are spec-written scenes whose expected values come from the
 * spectrum module directly, so this checks the importer's plumbing (which
 * parameter becomes which spectrum, the defaults, the env-map paths) on top of
 * the conversions the unit tests pin.
 */

import { BlackbodySpectrum, PiecewiseLinearSpectrum, Spectra, latlongMapToWorld, runPbrtScene, spectrumToRGB, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

const kBase = "/tests/gpu/assets/pbrt";

async function load(device: Device, name: string): Promise<Scene> {
    return runPbrtScene(device, await (await fetch(`${kBase}/${name}`)).text(), kBase);
}

/** Unpolarized conductor Fresnel at normal incidence. */
function fresnelNormal(eta: number, k: number): number {
    return ((eta - 1) ** 2 + k * k) / ((eta + 1) ** 2 + k * k);
}

gpuTest("PbrtSpectra.namedSampledAndBlackbodyValuesBecomeRGB", async ({ device }) => {
    const scene = await load(device, "spectra.pbrt");

    // Conductor: base colour is the normal-incidence Fresnel of gold's eta/k.
    const eta = spectrumToRGB(Spectra.getNamedSpectrum("metal-Au-eta")!);
    const k = spectrumToRGB(Spectra.getNamedSpectrum("metal-Au-k")!);
    const gold = [fresnelNormal(eta.x, k.x), fresnelNormal(eta.y, k.y), fresnelNormal(eta.z, k.z)];
    const conductor = scene.getMaterial(0);
    const base = conductor.basic.baseColor!;
    console.error(`# gold base colour ${[base.x, base.y, base.z].map((v) => v.toFixed(4)).join()} (expect ${gold.map((v) => v.toFixed(4)).join()})`);
    expectClose(base.x, gold[0]!, 1e-3, "gold red");
    expectClose(base.y, gold[1]!, 1e-3, "gold green");
    expectClose(base.z, gold[2]!, 1e-3, "gold blue");
    expectEq(base.x > base.z, true, "gold is warm");

    // Sampled reflectance spectrum.
    const reflectance = spectrumToRGB(new PiecewiseLinearSpectrum([400, 550, 700], [0.2, 0.5, 0.8]));
    const diffuse = scene.getMaterial(1).basic.baseColor!;
    expectClose(diffuse.x, reflectance.x, 1e-3, "sampled reflectance red");
    expectClose(diffuse.z, reflectance.z, 1e-3, "sampled reflectance blue");

    // Blackbody distant light, times its scale.
    const blackbody = spectrumToRGB(new BlackbodySpectrum(5000));
    const light = scene.getLight(0);
    expectClose(light.intensity.x, 2 * blackbody.x, 1e-4, "blackbody red");
    expectClose(light.intensity.z, 2 * blackbody.z, 1e-4, "blackbody blue");

    // Constant infinite light: a one-pixel env map holding D65 in RGB.
    const env = scene.getEnvMap();
    expectEq(env !== null, true, "the constant infinite light became an env map");
    const texel = new Float32Array((await device.renderContext.readTextureSubresource(env!.texture)).buffer);
    const d65 = spectrumToRGB(Spectra.getNamedSpectrum("stdillum-D65")!);
    console.error(`# D65 env texel ${[texel[0], texel[1], texel[2]].map((v) => v!.toFixed(4)).join()}`);
    for (let c = 0; c < 3; c++) expectClose(texel[c]!, [d65.x, d65.y, d65.z][c]!, 1e-5, `D65 channel ${c}`);
});

gpuTest("PbrtSpectra.octahedralEnvMapIsConvertedToLatLong", async ({ device }) => {
    const scene = await load(device, "octahedral-env.pbrt");
    const env = scene.getEnvMap()!;
    expectEq(env.intensity, 3, "scale -> intensity");
    const width = env.texture.width;
    const height = env.texture.height;
    expectEq(width, 32, "16x16 octahedral -> 32x16 lat-long");
    expectEq(height, 16, "lat-long height");
    const data = new Float32Array((await device.renderContext.readTextureSubresource(env.texture)).buffer);
    let upper = 0;
    let lower = 0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const dir = latlongMapToWorld((x + 0.5) / width, (y + 0.5) / height);
            if (Math.abs(dir[2]) < 0.45) continue; // bilinear blend at the horizon
            const i = (y * width + x) * 4;
            const expected = dir[2] > 0 ? [2, 1, 0.5] : [0.1, 0.2, 0.4];
            for (let c = 0; c < 3; c++) expectClose(data[i + c]!, expected[c]!, 1e-4, `hemisphere colour at ${x},${y}`);
            if (dir[2] > 0) upper++;
            else lower++;
        }
    }
    console.error(`# octahedral env: ${upper} upper and ${lower} lower texels checked`);
    expectEq(upper > 20 && lower > 20, true, "both hemispheres are present");
});
