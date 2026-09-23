/**
 * Full-scene OBJ import through the assimp importer (AssimpImporter's OBJ
 * mode) and the spec-gloss shading model it selects.
 *
 * Natively an OBJ/MTL material becomes a StandardMaterial in SpecGloss mode:
 * Kd is the diffuse colour, Ks the specular colour, and the Phong exponent Ns
 * turns into glossiness 1 - sqrt(2 / (Ns + 2)). The fixture's Ns = 30 gives
 * glossiness 0.75 exactly, and with Ks = 0.04 the result is mathematically the
 * metal-rough material (base Kd, roughness 0.25, no metal, IoR 1.5 so F0 =
 * 0.04). The G-buffer's material channels must therefore agree between the
 * two, which pins the importer, the shading-model bit, the scene define and
 * the shader's spec-gloss branch together.
 */

import { MaterialType, RenderGraph, SceneBuilderFlags, ShadingModel, createPass, initScripting, runSceneScript, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

const kBase = "/tests/gpu/assets/obj";
const size = 64;

async function loadScene(device: Device, name: string, flags?: number): Promise<Scene> {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch(`${kBase}/${name}`)).text();
    const scene = await runSceneScript(device, source, kBase, flags !== undefined ? { flags } : undefined);
    scene.camera.setAspectRatio(1);
    return scene;
}

/** Mean G-buffer material channels over the pixels that hit geometry. */
async function materialChannels(device: Device, scene: Scene): Promise<{ hits: number; diffuse: number[]; specRough: number[] }> {
    const graph = new RenderGraph(device, "ObjGBuffer");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("GBuffer.posW");
    graph.markOutput("GBuffer.diffuseOpacity");
    graph.markOutput("GBuffer.specRough");
    graph.onResize(size, size);
    graph.setScene(scene);
    await graph.init();
    const ctx = device.renderContext;
    graph.execute(ctx);
    const read = async (name: string) => new Float32Array((await ctx.readTextureSubresource(graph.getOutput(name)!)).buffer);
    const posW = await read("GBuffer.posW");
    const diffuse = await read("GBuffer.diffuseOpacity");
    const specRough = await read("GBuffer.specRough");
    const d = [0, 0, 0];
    const sr = [0, 0, 0, 0];
    let hits = 0;
    for (let i = 0; i < size * size; i++) {
        if (posW[i * 4 + 3] === 0) continue;
        hits++;
        for (let c = 0; c < 3; c++) d[c]! += diffuse[i * 4 + c]!;
        for (let c = 0; c < 4; c++) sr[c]! += specRough[i * 4 + c]!;
    }
    return { hits, diffuse: d.map((v) => v / Math.max(hits, 1)), specRough: sr.map((v) => v / Math.max(hits, 1)) };
}

gpuTest("AssimpObj.mtlBecomesASpecGlossStandardMaterial", async ({ device }) => {
    const scene = await loadScene(device, "obj-specgloss.pyscene");
    expectEq(scene.stats.materials, 1, "one material from the MTL");
    const material = scene.getMaterial(0);
    expectEq(material.header?.materialType ?? MaterialType.Standard, MaterialType.Standard, "OBJ materials are StandardMaterials");
    expectEq(material.basic.shadingModel, ShadingModel.SpecGloss, "OBJ mode selects spec-gloss");
    const spec = material.basic.specular!;
    const base = material.basic.baseColor!;
    console.error(`# OBJ material: base ${[base.x, base.y, base.z].join()} specular ${[spec.x, spec.y, spec.z, spec.w].join()}`);
    expectClose(base.x, 0.6, 1e-6, "Kd -> diffuse");
    expectClose(spec.x, 0.04, 1e-6, "Ks -> specular colour");
    expectClose(spec.w, 0.75, 1e-6, "Ns 30 -> glossiness 1 - sqrt(2 / 32)");
    expectEq(scene.getSceneDefines().get("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS"), "1", "the spec-gloss shader branch is compiled in");

    // The same MTL forced into metal-rough keeps the raw values, as natively.
    const forced = await loadScene(device, "obj-specgloss.pyscene", SceneBuilderFlags.UseMetalRoughMaterials);
    expectEq(forced.getMaterial(0).basic.shadingModel, ShadingModel.MetalRough, "UseMetalRoughMaterials overrides OBJ mode");
    expectEq(forced.getSceneDefines().get("MATERIAL_SYSTEM_HAS_SPEC_GLOSS_MATERIALS"), "0", "no spec-gloss material left");
});

gpuTest("AssimpObj.specGlossShadesLikeItsMetalRoughEquivalent", async ({ device }) => {
    const obj = await materialChannels(device, await loadScene(device, "obj-specgloss.pyscene"));
    const reference = await materialChannels(device, await loadScene(device, "metalrough-equivalent.pyscene"));
    console.error(`# spec-gloss:  ${obj.hits} hits, diffuse ${obj.diffuse.map((v) => v.toFixed(4)).join()} specRough ${obj.specRough.map((v) => v.toFixed(4)).join()}`);
    console.error(`# metal-rough: ${reference.hits} hits, diffuse ${reference.diffuse.map((v) => v.toFixed(4)).join()} specRough ${reference.specRough.map((v) => v.toFixed(4)).join()}`);
    expectEq(obj.hits > 1000, true, `the OBJ quad is on screen (${obj.hits} hits)`);
    expectEq(Math.abs(obj.hits - reference.hits) <= size, true, "both quads cover the same area");
    for (let c = 0; c < 3; c++) expectClose(obj.diffuse[c]!, reference.diffuse[c]!, 2e-3, `diffuse channel ${c}`);
    for (let c = 0; c < 4; c++) expectClose(obj.specRough[c]!, reference.specRough[c]!, 2e-3, `specular/roughness channel ${c}`);
    // And the roughness really is the converted glossiness, not the raw Ks.
    expectClose(obj.specRough[3]!, 0.25, 2e-3, "roughness = 1 - glossiness");
});

gpuTest("AssimpObj.tgaTexturesDecodeOnTheAssimpPath", async ({ device }) => {
    const scene = await loadScene(device, "obj-textured.pyscene");
    expectEq(scene.stats.textures, 1, "the TGA map_Kd reached the texture manager");
    const { hits, diffuse } = await materialChannels(device, scene);
    // Every texel is sRGB (255, 128, 0); the diffuse slot decodes it to linear.
    const toLinear = (v: number) => (v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4));
    const expected = [toLinear(1), toLinear(128 / 255), toLinear(0)];
    console.error(`# TGA diffuse over ${hits} hits: ${diffuse.map((v) => v.toFixed(4)).join()} (expect ${expected.map((v) => v.toFixed(4)).join()})`);
    for (let c = 0; c < 3; c++) expectClose(diffuse[c]!, expected[c]!, 5e-3, `texel channel ${c}`);
});

gpuTest("SpecGloss.pysceneMaterialMatchesItsMetalRoughEquivalent", async ({ device }) => {
    const scene = await loadScene(device, "specgloss-pyscene.pyscene");
    const material = scene.getMaterial("specGloss");
    expectEq(material.basic.shadingModel, ShadingModel.SpecGloss, "StandardMaterial(name, ShadingModel.SpecGloss)");
    expectClose(material.basic.specular!.y, 0.04, 1e-6, "roughness assignment ignored in spec-gloss mode");

    const specGloss = await materialChannels(device, scene);
    const reference = await materialChannels(device, await loadScene(device, "metalrough-equivalent.pyscene"));
    console.error(`# pyscene spec-gloss: specRough ${specGloss.specRough.map((v) => v.toFixed(4)).join()} vs ${reference.specRough.map((v) => v.toFixed(4)).join()}`);
    for (let c = 0; c < 3; c++) expectClose(specGloss.diffuse[c]!, reference.diffuse[c]!, 2e-3, `diffuse channel ${c}`);
    for (let c = 0; c < 4; c++) expectClose(specGloss.specRough[c]!, reference.specRough[c]!, 2e-3, `specular/roughness channel ${c}`);
});
