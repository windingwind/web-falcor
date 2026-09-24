/**
 * UsdUVTexture details tinyusdz drops (output channel selectors, sourceColorSpace,
 * the emissive value scale, UsdTransform2d), read from the layer text like native's
 * ConvertedInput, vs a native GBufferRT capture of usd-channels.pyscene. The stage
 * has no camera, so both add USDImporter's default one (the pyscene then selects its own).
 *
 * Native drops UsdTransform2d (its ConvertedTexTransform is updated before the
 * texture path is set), so the transformed quad's texcoords are checked against
 * the transform applied to native's untransformed ones. The emissive quad's
 * diffuse albedo checks its alpha-channel opacity (1 - specular transmission).
 *
 * Regenerate the oracle with:
 *   Falcor/build/linux-gcc/bin/Release/Mogwai --script tests/oracle/render-native-usd-channels.py --headless
 */

import { RenderGraph, createPass, initScripting, runSceneScript, usdTexCoordTransform } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("UsdChannels.matchesNativeOracle", async ({ device }) => {
    const [w, h] = [128, 64];
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/oracle/assets/usd-channels.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/oracle/assets");
    scene.camera.setAspectRatio(w / h);
    // Native's default: looking down (-1, -1, -1) at the stage center from 1.5 diagonals.
    const cam = scene.getCameras()[0]!;
    expectEq(cam.name, "Default", "the USD default camera comes first");
    for (const v of [cam.getPosition().x, cam.getPosition().y, cam.getPosition().z]) expectClose(v, 5.806893, 1e-5, "default camera position (native)");
    expectClose(cam.getFarPlane(), 26.820889, 1e-4, "default camera depth range (native)");
    expectEq(cam.getFocalLength(), 18, "default camera focal length");
    expectEq(scene.getMaterial(0)?.name, "ChannelsMat", "materials are named after their prims");
    expectEq(scene.getMaterial(1)?.basic.emissiveFactor, 3, "emissiveColor's texture scale -> emissive factor");

    const graph = new RenderGraph(device, "UsdChannels");
    graph.addPass(createPass(device, "GBufferRT", { useTraceRayInline: true, samplePattern: "Center" }), "GBufferRT");
    // The captures are half precision.
    const channels = [
        ["texC", 2, 3e-3],
        ["diffuseOpacity", 3, 2e-3],
        ["specRough", 3, 2e-3],
        ["emissive", 3, 4e-3],
    ] as const;
    graph.markOutput("GBufferRT.mask");
    graph.markOutput("GBufferRT.posW");
    for (const [c] of channels) graph.markOutput(`GBufferRT.${c}`);
    graph.onResize(w, h);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);

    const native = async (c: string) => (parseExr(await (await fetch(`/tests/oracle/out-native/usd-channels.GBufferRT.${c}.0.exr`)).arrayBuffer(), 1015) as { data: Float32Array }).data;
    const read = async (c: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(`GBufferRT.${c}`)!)).buffer);
    const natMask = await native("mask");
    const natPos = await native("posW");
    const webMask = await read("mask");
    const webPos = await read("posW");
    const transform = usdTexCoordTransform({ scale: [2, 0.5], rotation: 30, translation: [0.25, 0.1] });

    // Quads by world x: channels, emissive (alpha opacity), transformed.
    const quadOf = (ni: number) => (natPos[ni]! < -1.1 ? 0 : natPos[ni]! > 1.1 ? 2 : 1);
    const hits = [0, 0, 0];
    let maskMismatch = 0;
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
            // Native EXR captures are bottom-up.
            const ni = ((h - 1 - y) * w + x) * 4;
            if (natMask[ni] !== 0) hits[quadOf(ni)]!++;
            // The transformed quad is alpha-tested at other texcoords.
            if ((natMask[ni] !== 0 && quadOf(ni) === 2) || webPos[(y * w + x) * 4]! > 1.1) continue;
            if ((natMask[ni] !== 0) !== (webMask[y * w + x] !== 0)) maskMismatch++;
        }
    }
    console.error(`# usdChannels hits: channels ${hits[0]}, emissive ${hits[1]}, transformed ${hits[2]}; mask mismatches ${maskMismatch}`);
    expectEq(hits.every((n) => n > 100), true, "all three quads are hit");
    expectEq(maskMismatch <= 4, true, `hit masks match, incl. alpha-tested texels (${maskMismatch})`);

    for (const [c, components, tol] of channels) {
        const web = await read(c);
        const webComponents = web.length / (w * h);
        const nat = await native(c);
        let max = 0;
        for (let y = 0; y < h; y++) {
            for (let x = 0; x < w; x++) {
                const ni = ((h - 1 - y) * w + x) * 4;
                const wi = (y * w + x) * webComponents;
                if (natMask[ni] === 0 || webMask[y * w + x] === 0) continue;
                let expected = [...nat.slice(ni, ni + components)];
                if (quadOf(ni) === 2) {
                    if (c !== "texC") continue;
                    // Native's texcoords are (s, -t).
                    expected = transform(nat[ni]!, -nat[ni + 1]!);
                }
                for (let k = 0; k < components; k++) max = Math.max(max, Math.abs(web[wi + k]! - expected[k]!));
            }
        }
        console.error(`# usdChannels.${c}: max=${max.toExponential(2)}`);
        expectEq(max < tol, true, `${c} max ${max}`);
    }
});
