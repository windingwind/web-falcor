/**
 * PixelInspectorPass: functional GPU test. Runs the upstream GBufferRT graph
 * over the cornell box, then direct-executes the inspector on the G-buffer
 * outputs and cross-checks the readback record against the input textures at
 * the selected pixel (posW/texC/mtlData exact, normals normalized, vbuffer
 * decode sane, luminance = Rec.709 of the hand-made linColor input).
 */

import { Properties, RenderData, ResourceFormat, initScripting, runGraphScript, runSceneScript, type Texture } from "@web-falcor/falcor";
import { PixelInspectorPass } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("PixelInspector.matchesGBufferInputs", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const graphSource = await (await fetch("/Falcor/tests/image_tests/renderpasses/graphs/GBufferRT.py")).text();
    const [graph] = await runGraphScript(device, graphSource);

    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);

    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);

    const tex = (name: string): Texture => {
        const t = graph!.getOutput(`GBufferRT.${name}`);
        if (!t) throw new Error(`missing graph output ${name}`);
        return t;
    };

    // Hand-made 1x1 linColor input pins the scaled-coordinate path and luminance.
    const linColor = device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0.25, 0.5, 0.75, 1]));

    const inputs = new Map([
        ["posW", tex("posW")],
        ["normW", tex("normW")],
        ["tangentW", tex("tangentW")],
        ["faceNormalW", tex("faceNormalW")],
        ["texC", tex("texC")],
        ["texGrads", tex("texGrads")],
        ["mtlData", tex("mtlData")],
        ["linColor", linColor],
        ["vbuffer", tex("vbuffer")],
    ]);

    const pass = new PixelInspectorPass(device, new Properties({}));
    pass.setScene(scene);
    pass.setScaleInputsToWindow(true); // 1x1 linColor scales to (0,0); full-res inputs are unaffected
    const px = 128;
    const py = 140;
    pass.setCursorPosition((px + 0.5) / size, (py + 0.5) / size);
    pass.execute(ctx, new RenderData(inputs, [size, size]));
    const data = await pass.readPixelData();

    // Reference values read straight from the input textures.
    const posW = new Float32Array((await ctx.readTextureSubresource(tex("posW"))).buffer);
    const normW = new Float32Array((await ctx.readTextureSubresource(tex("normW"))).buffer);
    const texC = new Float32Array((await ctx.readTextureSubresource(tex("texC"))).buffer);
    const mtl = new Uint32Array((await ctx.readTextureSubresource(tex("mtlData"))).buffer);
    const idx4 = (py * size + px) * 4;
    const texCStride = texC.length / (size * size); // RG32Float may read back as 2 or 4 channels
    expectEq(posW[idx4 + 3]! !== 0, true, "selected pixel hits geometry");

    for (let c = 0; c < 3; c++) {
        expectEq(Math.abs(data.posW[c]! - posW[idx4 + c]!) < 1e-6, true, `posW[${c}] ${data.posW[c]} == ${posW[idx4 + c]}`);
    }
    // Shading normal is normalized in the kernel; cornell normals are already unit length.
    const nLen = Math.hypot(...data.normal);
    expectEq(Math.abs(nLen - 1) < 1e-4, true, `normal unit length ${nLen}`);
    let nDot = 0;
    for (let c = 0; c < 3; c++) nDot += data.normal[c]! * normW[idx4 + c]!;
    expectEq(nDot > 0.999, true, `normal matches normW (dot ${nDot})`);

    const tcBase = (py * size + px) * texCStride;
    expectEq(Math.abs(data.texCoord[0]! - texC[tcBase]!) < 1e-6, true, `texCoord.x ${data.texCoord[0]} == ${texC[tcBase]}`);
    expectEq(Math.abs(data.texCoord[1]! - texC[tcBase + 1]!) < 1e-6, true, `texCoord.y ${data.texCoord[1]} == ${texC[tcBase + 1]}`);

    expectEq(data.materialID, mtl[idx4]!, `materialID ${data.materialID} == ${mtl[idx4]}`);
    expectEq(data.frontFacing, 1, "front facing");

    // View vector: unit length, pointing from the surface toward the camera.
    const vLen = Math.hypot(...data.view);
    expectEq(Math.abs(vLen - 1) < 1e-4, true, `view unit length ${vLen}`);
    let vDotN = 0;
    for (let c = 0; c < 3; c++) vDotN += data.view[c]! * data.faceNormal[c]!;
    expectEq(vDotN > 0, true, `view faces the surface (dot ${vDotN})`);

    // Material instance properties: plausible ranges + guide normal near shading normal.
    expectEq(data.roughness >= 0 && data.roughness <= 1, true, `roughness ${data.roughness}`);
    let gDotN = 0;
    for (let c = 0; c < 3; c++) gDotN += data.guideNormal[c]! * data.normal[c]!;
    expectEq(gDotN > 0.99, true, `guideNormal matches normal (dot ${gDotN})`);
    const albedoOk = data.diffuseReflectionAlbedo.every((x) => x >= 0 && x <= 1);
    expectEq(albedoOk, true, `diffuse albedo ${data.diffuseReflectionAlbedo}`);
    // sd.IoR is the exterior medium IoR (air = 1.0).
    expectEq(data.IoR >= 1 && data.IoR < 3, true, `IoR ${data.IoR}`);

    // linColor (1x1, scaled coords) + Rec.709 luminance.
    expectEq(Math.abs(data.linearColor[0]! - 0.25) < 1e-6, true, `linearColor.r ${data.linearColor[0]}`);
    expectEq(Math.abs(data.linearColor[1]! - 0.5) < 1e-6, true, `linearColor.g ${data.linearColor[1]}`);
    expectEq(Math.abs(data.linearColor[2]! - 0.75) < 1e-6, true, `linearColor.b ${data.linearColor[2]}`);
    const lum = 0.2126 * 0.25 + 0.7152 * 0.5 + 0.0722 * 0.75;
    expectEq(Math.abs(data.luminance - lum) < 1e-5, true, `luminance ${data.luminance} == ${lum}`);
    // outColor unbound: dummy black.
    expectEq(data.outputColor.every((x) => x === 0), true, `outputColor dummy ${data.outputColor}`);

    // V-buffer decode: triangle hit with sane indices and barycentrics.
    expectEq(data.hitType, 1, `hitType ${data.hitType} == Triangle`);
    expectEq(data.instanceID !== 0xffffffff, true, "instanceID valid");
    expectEq(data.instanceID < 100 && data.primitiveIndex < 100000, true, `ids sane (${data.instanceID}, ${data.primitiveIndex})`);
    const [bx, by] = data.barycentrics;
    expectEq(bx! >= 0 && by! >= 0 && bx! + by! <= 1 + 1e-6, true, `barycentrics (${bx}, ${by})`);

    console.error(
        `# pixel-inspector: posW=(${data.posW.map((x) => x.toFixed(3)).join(",")}) mtl=${data.materialID} rough=${data.roughness.toFixed(3)} ` +
            `inst=${data.instanceID} prim=${data.primitiveIndex} lum=${data.luminance.toFixed(4)}`,
    );
});
