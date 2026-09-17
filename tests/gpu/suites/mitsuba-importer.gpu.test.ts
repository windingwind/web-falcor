/**
 * Mitsuba 3 XML importer (Scene/Importer/MitsubaImporter), mirroring upstream
 * Falcor's MitsubaImporter plugin.
 *
 * The media drop ships no Mitsuba content, so the fixtures are scenes written
 * to the format's spec that need no external meshes, which also makes them
 * exactly checkable:
 *   • constant-env: one diffuse rectangle under a uniform environment. An
 *     unoccluded Lambertian surface reflects exactly albedo * Le, so the
 *     rendered radiance has a closed form.
 *   • features: every built-in shape plus the BSDF types, an area emitter, an
 *     interior medium and the transform operations, checked against the values
 *     upstream's builder derives from them.
 */

import { RenderGraph, createPass, runMitsubaScene, MaterialType, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

const kBase = "/tests/gpu/assets/mitsuba";

async function loadScene(device: Device, name: string): Promise<Scene> {
    const source = await (await fetch(`${kBase}/${name}`)).text();
    return runMitsubaScene(device, source, kBase);
}

/** Renders to linear radiance (no tone mapping) and returns RGBA32F pixels. */
async function renderLinear(device: Device, scene: Scene, size: number, frames: number): Promise<Float32Array> {
    scene.camera.setAspectRatio(1);
    const graph = new RenderGraph(device, "Mitsuba");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "LightBVH" }), "PathTracer");
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("PathTracer.color", "Accumulate.input");
    graph.markOutput("Accumulate.output");
    graph.setScene(scene);
    await graph.init();
    for (let f = 0; f < frames; f++) graph.execute(device.renderContext);
    return new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("Accumulate.output")!)).buffer);
}

gpuTest("MitsubaImporter.diffuseUnderConstantEnvMatchesAlbedoTimesRadiance", async ({ device }) => {
    const scene = await loadScene(device, "constant-env.xml");
    expectEq(scene.stats.materials, 1, "one material");
    expectEq(scene.getMaterial(0).header?.materialType, MaterialType.PBRTDiffuse, "diffuse -> PBRTDiffuseMaterial");

    // fov 45 on a square film: upstream derives the focal length from the film
    // *width* (24 / height * width = 24 here).
    const expectedFocal = 24 / (2 * Math.tan(((45 / 2) * Math.PI) / 180));
    expectClose(scene.camera.getFocalLength(), expectedFocal, 1e-4, "focal length from fov and film size");
    const position = scene.camera.getPosition();
    expectClose(Math.hypot(position.x, position.y, position.z - 4), 0, 1e-5, "camera at the lookat origin");

    const size = 96;
    const px = await renderLinear(device, scene, size, 256);
    // Average the middle of the rectangle, which fills the frame centre.
    const rgb = [0, 0, 0];
    let n = 0;
    for (let y = size * 0.35; y < size * 0.65; y++) {
        for (let x = size * 0.35; x < size * 0.65; x++) {
            const i = ((y | 0) * size + (x | 0)) * 4;
            for (let c = 0; c < 3; c++) rgb[c]! += px[i + c]!;
            n++;
        }
    }
    const mean = rgb.map((v) => v / n);
    const expected = [0.8 * 0.5, 0.6 * 0.25, 0.4 * 0.75];
    console.error(`# Mitsuba constant env: mean ${mean.map((v) => v.toFixed(4)).join(", ")} (expect ${expected.map((v) => v.toFixed(4)).join(", ")})`);
    for (let c = 0; c < 3; c++) expectClose(mean[c]!, expected[c]!, 0.02 * expected[c]! + 2e-3, `channel ${c} equals albedo * Le`);
});

gpuTest("MitsubaImporter.buildsEveryPluginTheUpstreamImporterMaps", async ({ device }) => {
    const scene = await loadScene(device, "features.xml");

    // Five shapes, five materials (named after their BSDF, as upstream does).
    expectEq(scene.stats.instances, 5, "one instance per shape");
    expectEq(scene.stats.materials, 5, "one material per shape");

    const byName = (name: string) => scene.getMaterial(name);
    expectEq(byName("grey").header?.materialType, MaterialType.PBRTDiffuse, "diffuse -> PBRTDiffuse");
    expectClose(byName("grey").basic.baseColor?.x ?? 0, 0.35, 1e-6, "diffuse reflectance -> base color");

    // roughconductor: eta -> base color, k -> transmission color, roughness = alpha.
    const conductor = byName("metal");
    expectEq(conductor.header?.materialType, MaterialType.PBRTConductor, "roughconductor -> PBRTConductor");
    expectClose(conductor.basic.baseColor?.y ?? 0, 0.92, 1e-6, "conductor eta");
    expectClose(conductor.basic.transmission?.x ?? 0, 3.9, 1e-6, "conductor k");
    expectClose(conductor.basic.specular?.y ?? 0, 0.25, 1e-6, "conductor roughness = alpha");
    expectEq(conductor.header?.doubleSided, true, "conductors are double sided");

    // dielectric: ior = int_ior / ext_ior, with water and air from the IOR table.
    const glass = byName("glass");
    expectEq(glass.header?.materialType, MaterialType.Standard, "dielectric -> StandardMaterial");
    expectClose(glass.header?.ior ?? 0, 1.333 / 1.000277, 1e-6, "ior from the named IOR table");
    expectClose(glass.basic.specularTransmission ?? 0, 1, 1e-6, "dielectrics transmit");
    expectClose(glass.basic.volumeScattering?.x ?? 0, 0.5, 1e-6, "interior medium sigma_s");
    expectClose(glass.basic.volumeAbsorption?.z ?? 0, 0.06, 1e-6, "interior medium sigma_a");

    // twosided(roughplastic): roughness = sqrt(alpha), and the inner material is double sided.
    const plastic = byName("plastic");
    expectEq(plastic.header?.materialType, MaterialType.Standard, "roughplastic -> StandardMaterial");
    expectEq(plastic.header?.doubleSided, true, "twosided marks the inner material");
    expectClose(plastic.basic.specular?.y ?? 0, Math.sqrt(0.16), 1e-6, "roughness = sqrt(alpha)");
    expectClose(plastic.basic.baseColor?.y ?? 0, 0.5, 1e-6, "plastic diffuse reflectance");
    expectClose(plastic.header?.ior ?? 0, 1.49 / 1.000277, 1e-6, "polypropylene default ior");

    // The area emitter splits radiance into a unit colour and a factor.
    const light = byName("light");
    expectEq(light.header?.emissive, true, "area emitter -> emissive material");
    expectClose(light.basic.emissiveFactor ?? 0, 9, 1e-6, "emissive factor = max(radiance)");
    expectClose(light.basic.emissive?.y ?? 0, 6 / 9, 1e-6, "emissive color = radiance / factor");
    expectClose(light.basic.baseColor?.x ?? 0, 0.78, 1e-6, "the diffuse base color carries over");

    // The sensor's own parameters.
    expectClose(scene.camera.getFocalDistance(), 7, 1e-6, "focus_distance");
    expectClose(scene.camera.getApertureRadius(), 0.1, 1e-6, "aperture_radius");
    const target = scene.camera.getTarget();
    const position = scene.camera.getPosition();
    console.error(`# Mitsuba features: camera at ${[position.x, position.y, position.z].map((v) => v.toFixed(3)).join(", ")}`);
    expectClose(Math.hypot(position.x, position.y - 1, position.z - 6), 0, 1e-5, "camera at the lookat origin");
    // The lookat direction must survive the sensor's z flip.
    expectClose(target.z - position.z < 0 ? 0 : 1, 0, 1e-9, "camera looks toward the scene");
});

gpuTest("MitsubaImporter.upgradesLegacyScenesAndComposesTransforms", async ({ device }) => {
    const scene = await loadScene(device, "transforms.xml");
    // Both `toWorld` and `<lookAt>` are pre-2.0.0 spellings: if the upgrade pass
    // had not rewritten them, the shapes would sit at the origin and the camera
    // would keep its default pose.
    expectEq(scene.stats.instances, 2, "two cubes");
    const position = scene.camera.getPosition();
    expectClose(Math.hypot(position.x, position.y, position.z - 10), 0, 1e-5, "camera from the legacy <lookAt>");

    // Cube 1: size-2 cube scaled by 0.5 then moved to x = 3 -> x in [2.5, 3.5].
    // Cube 2: scaled by 0.25 through <matrix> then moved to y = -2.
    const bounds = scene.worldBounds!;
    console.error(`# Mitsuba transforms: bounds ${bounds.min.map((v) => v.toFixed(3)).join(", ")} .. ${bounds.max.map((v) => v.toFixed(3)).join(", ")}`);
    const expectedMin = [-0.25, -2.25, -0.5];
    const expectedMax = [3.5, 0.5, 0.5];
    for (let c = 0; c < 3; c++) {
        expectClose(bounds.min[c]!, expectedMin[c]!, 1e-5, `world bounds min ${c}`);
        expectClose(bounds.max[c]!, expectedMax[c]!, 1e-5, `world bounds max ${c}`);
    }
});

gpuTest("MitsubaImporter.rejectsTheTagsUpstreamLeavesUnimplemented", async ({ device }) => {
    // Upstream's parser throws "not implemented" for these four, so scenes using
    // them fail here in the same way rather than importing something different.
    for (const tag of ["default", "include", "alias", "path"]) {
        const source = `<scene version="3.0.0"><${tag} name="x" value="1"/></scene>`;
        let threw = "";
        try {
            await runMitsubaScene(device, source, kBase);
        } catch (e) {
            threw = String(e);
        }
        expectEq(threw.includes(`<${tag}> is not implemented`), true, `<${tag}> is rejected (got ${threw || "no error"})`);
    }

    // A missing version attribute is an error too.
    let threw = "";
    try {
        await runMitsubaScene(device, "<scene><shape type=\"cube\"/></scene>", kBase);
    } catch (e) {
        threw = String(e);
    }
    expectEq(threw.includes("missing version attribute"), true, `the version attribute is required (got ${threw || "no error"})`);
});

gpuTest("MitsubaImporter.checkerboardTextureReachesTheSurface", async ({ device }) => {
    const scene = await loadScene(device, "checkerboard.xml");
    const size = 96;
    const px = await renderLinear(device, scene, size, 128);

    // Mean radiance of each quadrant of the rectangle, sampled away from the
    // checker seam and the silhouette.
    const quadrant = (qx: number, qy: number): number[] => {
        const rgb = [0, 0, 0];
        let n = 0;
        const x0 = size * (qx ? 0.56 : 0.28);
        const y0 = size * (qy ? 0.56 : 0.28);
        for (let y = y0; y < y0 + size * 0.16; y++) {
            for (let x = x0; x < x0 + size * 0.16; x++) {
                const i = ((y | 0) * size + (x | 0)) * 4;
                for (let c = 0; c < 3; c++) rgb[c]! += px[i + c]!;
                n++;
            }
        }
        return rgb.map((v) => v / n);
    };
    const tl = quadrant(0, 0);
    const tr = quadrant(1, 0);
    const bl = quadrant(0, 1);
    const br = quadrant(1, 1);
    console.error(`# Mitsuba checkerboard: tl ${tl.map((v) => v.toFixed(3)).join()} tr ${tr.map((v) => v.toFixed(3)).join()} bl ${bl.map((v) => v.toFixed(3)).join()} br ${br.map((v) => v.toFixed(3)).join()}`);

    // Under radiance 1 the surface reflects its own albedo. The 8-bit material
    // texture array quantizes the colours (§9), hence the 0.01 tolerance.
    const red = [0.9, 0.1, 0.1];
    const blue = [0.1, 0.1, 0.9];
    const isColor = (v: number[], c: number[]) => v.every((x, i) => Math.abs(x - c[i]!) < 0.015);
    // Which diagonal carries which colour depends on the uv orientation; what
    // must hold is that diagonals agree and the two colours both appear.
    expectEq(isColor(tl, red) === isColor(br, red), true, "diagonal quadrants share a colour");
    expectEq(isColor(tr, blue) === isColor(bl, blue), true, "the other diagonal shares the other colour");
    expectEq(isColor(tl, red) || isColor(tl, blue), true, `top-left is one of the checker colours (${tl.join()})`);
    expectEq(isColor(tr, red) || isColor(tr, blue), true, `top-right is one of the checker colours (${tr.join()})`);
    expectEq(isColor(tl, red) !== isColor(tr, red), true, "neighbouring quadrants differ");
});

gpuTest("MitsubaImporter.envmapEmitterLoadsAndScales", async ({ device }) => {
    const source = await (await fetch(`${kBase}/envmap.xml`)).text();
    const size = 64;
    // The same environment at two scales: the port resolves the filename against
    // the media root, and `scale` must land on the env map's intensity.
    const render = async (scale: string) => {
        const scene = await runMitsubaScene(device, source.replace('name="scale" value="1"', `name="scale" value="${scale}"`), "/Falcor/media");
        return renderLinear(device, scene, size, 8);
    };
    const single = await render("1");
    const double = await render("2");

    let mean = 0;
    let worstRatio = 0;
    let compared = 0;
    for (let i = 0; i < size * size; i++) {
        for (let c = 0; c < 3; c++) {
            const a = single[i * 4 + c]!;
            mean += a;
            if (a > 1e-3) {
                worstRatio = Math.max(worstRatio, Math.abs(double[i * 4 + c]! / a - 2));
                compared++;
            }
        }
    }
    mean /= size * size * 3;
    console.error(`# Mitsuba envmap: mean radiance ${mean.toFixed(4)}, worst |ratio - 2| = ${worstRatio.toExponential(2)} over ${compared} channels`);
    expectEq(mean > 0.01, true, `the environment map is visible (mean ${mean})`);
    expectEq(compared > size * size, true, `most of the frame carries radiance (${compared} channels)`);
    expectClose(worstRatio, 0, 1e-3, "scale multiplies the environment radiance");
});

gpuTest("MitsubaImporter.bitmapTextureBindsToTheDiffuseSlot", async ({ device }) => {
    const source = await (await fetch(`${kBase}/bitmap.xml`)).text();
    const scene = await runMitsubaScene(device, source, "/Falcor/media");
    expectEq(scene.stats.textures >= 1, true, "the bitmap reached the texture manager");

    // Mean linear albedo of the source image, for comparison with the render.
    const bitmap = await createImageBitmap(await (await fetch("/Falcor/media/test_images/monalisa.jpg")).blob(), { colorSpaceConversion: "none" });
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const g = canvas.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bitmap, 0, 0);
    const texels = g.getImageData(0, 0, bitmap.width, bitmap.height).data;
    const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    const textureMean = [0, 0, 0];
    for (let i = 0; i < texels.length; i += 4) {
        for (let c = 0; c < 3; c++) textureMean[c]! += toLinear(texels[i + c]! / 255);
    }
    for (let c = 0; c < 3; c++) textureMean[c]! /= texels.length / 4;

    const size = 96;
    const px = await renderLinear(device, scene, size, 96);
    // The rectangle nearly fills the frame at fov 30 (its half-extent is 1, the
    // frame's 4*tan(15deg) = 1.072), so average all of it: the environment has
    // radiance 1 and no albedo reaches that, which separates the background.
    const rendered = [0, 0, 0];
    let n = 0;
    let variance = 0;
    const samples: number[] = [];
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const i = (y * size + x) * 4;
            if (px[i]! > 0.95 && px[i + 1]! > 0.95 && px[i + 2]! > 0.95) continue; // background
            for (let c = 0; c < 3; c++) rendered[c]! += px[i + c]!;
            samples.push(px[i + 1]!);
            n++;
        }
    }
    for (let c = 0; c < 3; c++) rendered[c]! /= n;
    const meanG = rendered[1]!;
    for (const v of samples) variance += (v - meanG) * (v - meanG);
    variance /= samples.length;

    console.error(`# Mitsuba bitmap: ${n} surface pixels, rendered ${rendered.map((v) => v.toFixed(3)).join()} vs texture ${textureMean.map((v) => v.toFixed(3)).join()}, green std ${Math.sqrt(variance).toFixed(3)}`);
    expectEq(n > size * size * 0.8, true, `the textured rectangle fills the frame (${n} pixels)`);
    // A constant base colour would show zero spatial variation.
    expectEq(Math.sqrt(variance) > 0.02, true, `the texture varies across the surface (std ${Math.sqrt(variance)})`);
    for (let c = 0; c < 3; c++) {
        expectClose(rendered[c]!, textureMean[c]!, 0.1 * textureMean[c]! + 0.01, `channel ${c} matches the texture's mean albedo`);
    }
});

gpuTest("MitsubaImporter.objShapeLoadsAndTransforms", async ({ device }) => {
    const scene = await loadScene(device, "objshape.xml");
    expectEq(scene.stats.instances, 1, "one mesh instance");
    // The quad spans [-1, 1] in x and y; scaled by (2, 0.5, 1) and moved to z = 2.
    const bounds = scene.worldBounds!;
    console.error(`# Mitsuba obj shape: bounds ${bounds.min.map((v) => v.toFixed(3)).join(", ")} .. ${bounds.max.map((v) => v.toFixed(3)).join(", ")}`);
    const expectedMin = [-2, -0.5, 2];
    const expectedMax = [2, 0.5, 2];
    for (let c = 0; c < 3; c++) {
        expectClose(bounds.min[c]!, expectedMin[c]!, 1e-4, `world bounds min ${c}`);
        expectClose(bounds.max[c]!, expectedMax[c]!, 1e-4, `world bounds max ${c}`);
    }
});
