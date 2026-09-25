/**
 * Transplanted FalcorTest GPU tests Scene/EnvMapTests (the importance map is a square power-of-two
 * texture with a full mip chain) and Scene/Material/MERLFileTests (name, data size and the albedo
 * LUT, which comes from the cached .dds beside the BRDF as native's prepareAlbedoLUT loads it).
 */

import { EnvMap, EnvMapSampler, MaterialType, Scene, kMERLAlbedoLUTSize, kMERLSampleCount, loadMERLBinary, loadRGLFile } from "@web-falcor/falcor";
import { gpuTest, expectEq, SkipError } from "../../harness/registry.js";

gpuTest("FalcorTest.EnvMap", async ({ device }) => {
    const envMap = await EnvMap.createFromUrl(device, "/Falcor/media/test_scenes/envmaps/20050806-03_hd.hdr");
    const sampler = new EnvMapSampler(device, device.renderContext, envMap);
    const map = sampler.getImportanceMap();
    const [w, h] = [map.width, map.height];
    expectEq(w > 0 && (w & (w - 1)) === 0, true, `power-of-two width (${w})`);
    expectEq(w, h, "square");
    expectEq(w, 1 << (map.mipCount - 1), "full mip chain");
});

gpuTest("FalcorTest.MERLFile", async ({ device }) => {
    const merl = await loadMERLBinary("/Falcor/media/test_scenes/materials/data/gray-lambert.binary");
    expectEq(merl.name, "gray-lambert", "name");
    expectEq(merl.data.length / 3, (90 * 90 * 360) / 2, "data size (float3 per bin)");
    expectEq(kMERLSampleCount, (90 * 90 * 360) / 2, "sample count");
    const lut = merl.albedoLUT!;
    expectEq(lut?.length, kMERLAlbedoLUTSize * 4, "albedo LUT size");
    let exact = true;
    for (let i = 0; i < kMERLAlbedoLUTSize; i++) exact &&= lut[i * 4] === 0.5 && lut[i * 4 + 1] === 0.5 && lut[i * 4 + 2] === 0.5;
    expectEq(exact, true, "every LUT entry is float3(0.5)");

    // The scene takes the cached table instead of integrating the BRDF.
    const scene = new Scene(device, [], [{ name: merl.name, basic: {}, merl, header: { materialType: MaterialType.MERL } }]);
    await scene.computeMeasuredAlbedoLUTs(device.renderContext);
    const buffers = (scene as unknown as { buffers: Record<string, { getBlob(): Promise<Uint8Array> }>; merlOffsets: Map<number, { lut: number }> });
    const blob = await buffers.buffers["materialBuffer0"]!.getBlob();
    const at = buffers.merlOffsets.get(0)!.lut;
    const table = new Float32Array(blob.buffer, blob.byteOffset + at, kMERLAlbedoLUTSize * 4);
    expectEq(table.every((v, i) => v === lut[i]), true, "scene LUT = cached table");
});

// RGLMaterial::prepareAlbedoLUT: a cached table (here set directly) replaces the integration.
gpuTest("RGLMaterial.cachedAlbedoLUT", async ({ device }) => {
    const url = "/Falcor/media/rgl/acrylic_felt_green_rgb.bsdf";
    if (!(await fetch(url, { method: "HEAD" })).ok) throw new SkipError("RGL measurements missing (npm run download:assets -- rgl)");
    const rgl = await loadRGLFile(url);
    rgl.albedoLUT = Float32Array.from({ length: kMERLAlbedoLUTSize * 4 }, (_, i) => (i % 4 === 3 ? 1 : 0.25 + i / 8192));
    const scene = new Scene(device, [], [{ name: rgl.name, basic: {}, rgl, header: { materialType: MaterialType.RGL } }]);
    await scene.computeMeasuredAlbedoLUTs(device.renderContext);
    const s = scene as unknown as { buffers: Record<string, { getBlob(): Promise<Uint8Array> }>; rglOffsets: Map<number, Record<string, number>> };
    const blob = await s.buffers["materialBuffer0"]!.getBlob();
    const table = new Float32Array(blob.buffer, blob.byteOffset + s.rglOffsets.get(0)!["albedoLUT"]! * 4, kMERLAlbedoLUTSize * 4);
    expectEq(table.every((v, i) => v === rgl.albedoLUT![i]), true, "scene LUT = cached table");
});
