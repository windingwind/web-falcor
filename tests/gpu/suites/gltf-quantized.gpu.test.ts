/**
 * KHR_mesh_quantization: geometry stored as normalized bytes and raw shorts
 * instead of floats, with the node transform (and KHR_texture_transform for the
 * uvs) putting it back in place.
 *
 * Ground truth is the same model's uncompressed variant: both are rendered and
 * compared, so the decode is checked against float geometry rather than itself.
 * The two files are separate exports (the quantized one is reordered and
 * re-welded), so the comparison is distributional rather than pixel-exact.
 *
 * Fetch both with: npm run download:assets -- gltf-variants
 */

import { GltfImporter, RenderGraph, createPass } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, SkipError } from "../harness/registry.js";

const size = 256;
const kDir = "/Falcor/media/gltf-variants/";

gpuTest("GltfQuantized.matchesTheUncompressedVariant", async ({ device }) => {
    if (!(await fetch(`${kDir}Duck-Quantized/Duck.gltf`, { method: "HEAD" })).ok) {
        throw new SkipError("Falcor/media/gltf-variants missing (npm run download:assets -- gltf-variants)");
    }

    const render = async (url: string) => {
        const scene = await GltfImporter.importFromUrl(device, url);
        scene.camera.setAspectRatio(1.0);
        const graph = new RenderGraph(device, "Quantized");
        graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBufferRT");
        graph.markOutput("GBufferRT.posW");
        graph.markOutput("GBufferRT.normW");
        graph.markOutput("GBufferRT.texC");
        graph.onResize(size, size);
        graph.setScene(scene);
        await graph.init();
        const ctx = device.renderContext;
        graph.execute(ctx);
        const read = async (name: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${name}`)!)).buffer);
        return { posW: await read("posW"), normW: await read("normW"), texC: await read("texC"), triangles: scene.stats.instances };
    };

    const reference = await render(`${kDir}Duck/Duck.gltf`);
    const quantized = await render(`${kDir}Duck-Quantized/Duck.gltf`);

    // The models share a camera, so the two renders line up pixel for pixel.
    // Compare distributions rather than extremes: at silhouette edges the two
    // models can hit different faces, which says nothing about the decode.
    let hitsReference = 0;
    let hitsQuantized = 0;
    let coverageMismatch = 0;
    const positionErrors: number[] = [];
    const normalErrors: number[] = [];
    const uvErrors: number[] = [];
    for (let i = 0; i < size * size; i++) {
        const a = reference.posW[i * 4 + 3]! !== 0;
        const b = quantized.posW[i * 4 + 3]! !== 0;
        if (a) hitsReference++;
        if (b) hitsQuantized++;
        if (a !== b) {
            coverageMismatch++;
            continue;
        }
        if (!a) continue;
        positionErrors.push(
            Math.hypot(
                reference.posW[i * 4]! - quantized.posW[i * 4]!,
                reference.posW[i * 4 + 1]! - quantized.posW[i * 4 + 1]!,
                reference.posW[i * 4 + 2]! - quantized.posW[i * 4 + 2]!,
            ),
        );
        const dot =
            reference.normW[i * 4]! * quantized.normW[i * 4]! +
            reference.normW[i * 4 + 1]! * quantized.normW[i * 4 + 1]! +
            reference.normW[i * 4 + 2]! * quantized.normW[i * 4 + 2]!;
        normalErrors.push((Math.acos(Math.min(1, Math.max(-1, dot))) * 180) / Math.PI);
        uvErrors.push(Math.max(Math.abs(reference.texC[i * 2]! - quantized.texC[i * 2]!), Math.abs(reference.texC[i * 2 + 1]! - quantized.texC[i * 2 + 1]!)));
    }
    const compared = positionErrors.length;
    const percentile = (values: number[], q: number) => {
        const sorted = [...values].sort((x, y) => x - y);
        return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] ?? 0;
    };

    console.error(
        `# quantized vs float: hits ${hitsQuantized}/${hitsReference}, coverage mismatch ${coverageMismatch}; ` +
            `|dposW| p50=${percentile(positionErrors, 0.5).toExponential(2)} p99=${percentile(positionErrors, 0.99).toExponential(2)}; ` +
            `normal p50=${percentile(normalErrors, 0.5).toFixed(2)} p99=${percentile(normalErrors, 0.99).toFixed(2)} deg; ` +
            `|duv| p50=${percentile(uvErrors, 0.5).toExponential(2)} p99=${percentile(uvErrors, 0.99).toExponential(2)}`,
    );
    // Pixels where the two models disagree beyond quantization: at a silhouette a
    // sub-quantization shift flips which triangle is hit, which says nothing
    // about the decode, so allow a small budget of them.
    let outliers = 0;
    for (let i = 0; i < compared; i++) {
        if (positionErrors[i]! > 5e-3 || normalErrors[i]! > 3 || uvErrors[i]! > 5e-3) outliers++;
    }
    console.error(
        `# quantized vs float: ${outliers}/${compared} pixels beyond the quantization budget; ` +
            `|dposW| p90=${percentile(positionErrors, 0.9).toExponential(2)}; normal p90=${percentile(normalErrors, 0.9).toFixed(2)} deg; |duv| p90=${percentile(uvErrors, 0.9).toExponential(2)}`,
    );
    expectEq(hitsReference > 1000, true, `the reference model is visible (${hitsReference} hits)`);
    expectEq(coverageMismatch < hitsReference * 0.02, true, `silhouettes agree (${coverageMismatch} mismatching pixels)`);
    // The Duck spans a couple of units; 16-bit positions quantize far below that.
    expectEq(percentile(positionErrors, 0.5) < 1e-3, true, `positions match the float model (median ${percentile(positionErrors, 0.5)})`);
    // Normals are normalized bytes: a fraction of a degree in the median.
    expectEq(percentile(normalErrors, 0.5) < 1, true, `normals match (median ${percentile(normalErrors, 0.5)} deg)`);
    expectEq(percentile(uvErrors, 0.5) < 1e-3, true, `uvs match after KHR_texture_transform (median ${percentile(uvErrors, 0.5)})`);
    // 90% of the surface must agree to within quantization. The variants are
    // separate exports (the quantized one is reordered and re-welded), so a few
    // percent of pixels legitimately land on slightly different geometry.
    expectEq(percentile(positionErrors, 0.9) < 5e-3, true, `positions agree over the surface (p90 ${percentile(positionErrors, 0.9)})`);
    expectEq(percentile(normalErrors, 0.9) < 3, true, `normals agree over the surface (p90 ${percentile(normalErrors, 0.9)} deg)`);
    expectEq(percentile(uvErrors, 0.9) < 5e-3, true, `uvs agree over the surface (p90 ${percentile(uvErrors, 0.9)})`);
    expectEq(outliers < compared * 0.05, true, `few disagreeing pixels (${outliers}/${compared})`);
    expectEq(compared > 1000, true, `enough compared pixels (${compared})`);
});
