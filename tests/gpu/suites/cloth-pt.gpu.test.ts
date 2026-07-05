/**
 * M7 material zoo: ClothMaterial through the static-dispatch factory
 * (WEBFALCOR_MTL_CLOTH) — path-traced cloth quad vs the native hardware-DXR
 * oracle. The scene mirrors tests/oracle/assets/oracle-pt-cloth.pyscene:
 * TriangleMesh.createQuad(1,1) rotated 90° about X into [0,1]^2 at z=0.
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-pt-cloth.py --headless
 */

import {
    LightType,
    MaterialType,
    RenderGraph,
    Scene,
    createPass,
    float2,
    float3,
    float4,
    float4x4,
    generateTangents,
    matrixFromTranslation,
    mulMat,
    type StaticVertex,
} from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("ClothPathTracer.matchesNativeOracle", async ({ device }) => {
    const size = 256;

    // TriangleMesh::createQuad(float2(1,1)): XZ plane, +Y normal.
    const n = new float3(0, 1, 0);
    const t0 = new float4(0, 0, 0, 0);
    const vertices: StaticVertex[] = [
        { position: new float3(-0.5, 0, -0.5), normal: n, tangent: t0, texCrd: new float2(0, 0) },
        { position: new float3(0.5, 0, -0.5), normal: n, tangent: t0, texCrd: new float2(1, 0) },
        { position: new float3(-0.5, 0, 0.5), normal: n, tangent: t0, texCrd: new float2(0, 1) },
        { position: new float3(0.5, 0, 0.5), normal: n, tangent: t0, texCrd: new float2(1, 1) },
    ];
    const indices = new Uint32Array([2, 1, 0, 1, 2, 3]);
    generateTangents(vertices, indices);

    // Transform(translation=(0.5,0.5,0), rotationEulerDeg=(90,0,0)).
    const rotX90 = new float4x4();
    rotX90.set(0, 0, 1);
    rotX90.set(1, 2, -1);
    rotX90.set(2, 1, 1);
    rotX90.set(3, 3, 1);
    const transform = mulMat(matrixFromTranslation(new float3(0.5, 0.5, 0)), rotX90);

    const scene = new Scene(
        device,
        [{ vertices, indices, materialID: 0, transform }],
        [
            {
                header: { materialType: MaterialType.Cloth },
                basic: {
                    baseColor: new float4(0.6, 0.3, 0.2, 1),
                    specular: new float4(0, 0.7, 0, 0), // ClothMaterial::setRoughness -> specular.g
                },
            },
        ],
        [{ type: LightType.Point, posW: new float3(0.5, 0.5, 1.5), intensity: new float3(3, 3, 3) }],
    );
    scene.camera.setPosition(new float3(0.5, 0.5, 2.0));
    scene.camera.setTarget(new float3(0.5, 0.5, -1.0));
    scene.camera.setAspectRatio(1.0);

    const graph = new RenderGraph(device, "ClothPTGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 3 }), "MinimalPathTracer");
    graph.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer");
    graph.markOutput("MinimalPathTracer.color");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);

    const web = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("MinimalPathTracer.color")!)).buffer);

    const res = await fetch("/tests/oracle/out-native/oracle-pt-cloth.MinimalPathTracer.color.0.exr");
    const { data, width, height } = parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    expectEq(width, size, "oracle resolution");

    let sum = 0;
    let refSum = 0;
    let badPixels = 0;
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const webIdx = (y * size + x) * 4;
            const natIdx = ((height - 1 - y) * width + x) * 4;
            let pixelMax = 0;
            for (let c = 0; c < 3; c++) {
                const d = Math.abs(web[webIdx + c]! - data[natIdx + c]!);
                sum += d;
                refSum += Math.abs(data[natIdx + c]!);
                pixelMax = Math.max(pixelMax, d);
            }
            if (pixelMax > 0.05) badPixels++;
        }
    }
    const mean = sum / (size * size * 3);
    const rel = sum / Math.max(refSum, 1e-6);
    console.error(`# clothPT: meanAbs=${mean.toExponential(2)} rel=${rel.toExponential(2)} bad=${badPixels}`);
    expectEq(mean < 5e-3, true, `radiance mean abs diff ${mean}`);
    expectEq(badPixels < size * 4, true, `bad pixels ${badPixels}`);
});
