/**
 * Material textures from DDS files (`material.loadTexture(slot, 'x.dds')`): every BC
 * format decodes on the GPU, as natively, before joining the RGBA8 texture arrays.
 * Checked against native's decode of the same files (the DDSReadTests reference
 * PNGs, stored bottom-up; native truncates where the blit rounds, hence ±1).
 */

import { ResourceBindFlags, ResourceFormat, decodePng, initScripting, runSceneScript } from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

gpuTest("MaterialDdsTexture.bcFormatsMatchNativeDecode", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    for (const name of ["BC1Unorm", "BC4Unorm", "BC7Unorm", "BC7UnormOdd"]) {
        const scene = await runSceneScript(
            device,
            `
m = StandardMaterial('M')
m.loadTexture(MaterialTextureSlot.Normal, '${name}.dds')
sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(float2(1.0, 1.0)), m))
`,
            "/Falcor/data/tests",
        );
        const tm = (scene as unknown as { lcTextureManager: { count: number; getSource(id: number): { bitmap: ImageBitmap } | undefined } }).lcTextureManager;
        expectEq(tm.count, 1, `${name}: texture loaded`);
        const bitmap = tm.getSource(0)!.bitmap;
        // Read back through a texture (a 2D canvas would premultiply alpha).
        const tex = device.createTexture2D(bitmap.width, bitmap.height, ResourceFormat.RGBA8Unorm, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        device.gpuDevice.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex.gpuTexture }, { width: bitmap.width, height: bitmap.height });
        const got = await device.renderContext.readTextureSubresource(tex, 0);
        const ref = await decodePng(new Uint8Array(await (await fetch(`/Falcor/data/tests/${name}-ref.png`)).arrayBuffer()));
        expectEq(`${bitmap.width}x${bitmap.height}`, `${ref.width}x${ref.height}`, `${name}: size`);
        let worst = 0;
        for (let y = 0; y < ref.height; y++)
            for (let x = 0; x < ref.width; x++)
                for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(got[(y * ref.width + x) * 4 + c]! - ref.data[((ref.height - 1 - y) * ref.width + x) * ref.channels + c]!));
        console.error(`# material-dds ${name}: ${bitmap.width}x${bitmap.height} worst=${worst}`);
        expectEq(worst <= 1, true, `${name}: worst difference from native ${worst}`);
    }
});
