/**
 * UsdPreviewSurface opacity through the USD importer, following native's
 * PreviewSurfaceConverter: uniform opacity becomes specular transmission,
 * opacity under an opacity threshold becomes an alpha cutout, and textured
 * opacity becomes a transmission texture. Every material is double-sided.
 */

import { AlphaMode, RenderGraph, createPass, initScripting, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("UsdOpacity.mapsLikePreviewSurfaceConverter", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/usd-opacity.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(1);
    expectEq(scene.stats.materials, 4, "four materials");

    const [transmit, cutout, textured, texCutout] = [0, 1, 2, 3].map((i) => scene.getMaterial(i));
    for (const m of [transmit!, cutout!, textured!, texCutout!]) expectEq(m.header?.doubleSided, true, "UsdPreviewSurface materials are double-sided");

    expectClose(transmit!.basic.specularTransmission ?? 0, 0.6, 1e-6, "uniform opacity 0.4 -> specular transmission 0.6");
    expectClose(transmit!.basic.baseColor!.w, 1, 1e-6, "no cutout without a threshold");

    expectClose(cutout!.basic.baseColor!.w, 0.3, 1e-6, "cutout opacity rides in the base colour alpha");
    expectClose(cutout!.header?.alphaThreshold ?? 0, 0.5, 1e-6, "opacityThreshold -> alpha threshold");
    expectEq(cutout!.basic.specularTransmission ?? 0, 0, "a cutout does not transmit");

    expectEq(textured!.basic.texTransmission !== undefined, true, "textured opacity -> transmission texture");
    expectClose(textured!.basic.specularTransmission ?? 0, 1, 1e-6, "with full specular transmission");

    expectEq(texCutout!.basic.texBaseColor !== undefined, true, "textured cutout opacity is packed into a base colour texture");
    expectClose(texCutout!.header?.alphaThreshold ?? 0, 0.7, 1e-6, "with the threshold");

    // Native's alpha test reads the base colour *texture's* alpha (the uniform
    // fallback is 1), so the untextured cutout stays visible there and here;
    // the textured one is cut wherever the checker's red channel is below its 0.7
    // threshold (the checker's reds are 0.55 and 0.90).
    const size = 96;
    const graph = new RenderGraph(device, "UsdOpacity");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center", useAlphaTest: true }), "GBuffer");
    graph.markOutput("GBuffer.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const posW = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBuffer.posW")!)).buffer);
    const counts = [0, 0, 0, 0];
    for (let i = 0; i < size * size; i++) {
        if (posW[i * 4 + 3] === 0) continue;
        const x = posW[i * 4]!;
        counts[x < -2 ? 0 : x < 0 ? 1 : x < 2 ? 2 : 3]!++;
    }
    // The fraction of the checker texture whose red channel passes the threshold.
    const bitmap = await createImageBitmap(await (await fetch("/tests/oracle/assets/oracle-usd-checker.png")).blob(), { colorSpaceConversion: "none" });
    const c2d = new OffscreenCanvas(bitmap.width, bitmap.height).getContext("2d")!;
    c2d.drawImage(bitmap, 0, 0);
    const texels = c2d.getImageData(0, 0, bitmap.width, bitmap.height).data;
    let kept = 0;
    for (let i = 0; i < texels.length; i += 4) if (texels[i]! / 255 >= 0.7) kept++;
    const keptFraction = kept / (texels.length / 4);
    const full = counts[0]!;
    console.error(`# USD opacity hits: transmit ${counts[0]}, uniform cutout ${counts[1]}, textured transmission ${counts[2]}, textured cutout ${counts[3]} (expect ~${Math.round(full * keptFraction)} = ${(keptFraction * 100).toFixed(0)}% of a full quad)`);
    expectEq(counts[0]! > 100 && counts[2]! > 100, true, "the transmissive quads are hit");
    expectEq(Math.abs(counts[1]! - full) <= full * 0.05, true, "the untextured cutout is not cut (native's alpha test reads the texture)");
    expectEq(Math.abs(counts[3]! - full * keptFraction) <= full * 0.1, true, `the textured cutout keeps only the passing texels (${counts[3]})`);
    expectEq(keptFraction > 0.1 && keptFraction < 0.9, true, "the checker really does cut part of the quad");
    void AlphaMode;
});
