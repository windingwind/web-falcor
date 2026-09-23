/**
 * Material textures share one texture array per colour space, sized to the largest
 * texture. Natively every texture has its own mip chain, so how a texture samples
 * cannot depend on which other textures the scene holds. This renders a minified,
 * grazing 1024² texture alone and again next to an off-screen 2048² texture (which
 * doubles the array) and requires the same result, for ray-cone and ray-differential
 * LODs (GBufferRT) and implicit LODs (GBufferRaster).
 */

import { initScripting, runGraphScript, runSceneScript, type Device } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

const scene = (withLargeTexture: boolean) => `
floor = StandardMaterial('Floor')
floor.loadTexture(MaterialTextureSlot.BaseColor, 'textures/bricks_height.png')
quad = TriangleMesh.createQuad(float2(1.0, 1.0))
sceneBuilder.addMeshInstance(sceneBuilder.addNode('floor', Transform(scaling=float3(8.0, 1.0, 8.0))), sceneBuilder.addTriangleMesh(quad, floor))
${
    withLargeTexture
        ? `other = StandardMaterial('Other')
other.loadTexture(MaterialTextureSlot.BaseColor, 'textures/checker_tile_base_color.png')
sceneBuilder.addMeshInstance(sceneBuilder.addNode('other', Transform(translation=float3(0.0, 0.0, 50.0))), sceneBuilder.addTriangleMesh(quad, other))`
        : ""
}
camera = Camera()
camera.position = float3(0.0, 0.6, -3.5)
camera.target = float3(0.0, 0.0, 0.0)
camera.up = float3(0.0, 1.0, 0.0)
sceneBuilder.addCamera(camera)
`;

async function render(device: Device, pass: string, props: string, withLargeTexture: boolean): Promise<Float32Array> {
    const [graph] = await runGraphScript(
        device,
        `
from falcor import *
g = RenderGraph('TextureArrayLOD')
g.addPass(createPass('${pass}', {'samplePattern': 'Center'${props}}), 'G')
g.markOutput('G.diffuseOpacity')
try: m.addGraph(g)
except NameError: None
`,
    );
    const s = await runSceneScript(device, scene(withLargeTexture), "/Falcor/media/test_scenes");
    s.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(s);
    graph!.execute(device.renderContext);
    const bytes = await device.renderContext.readTextureSubresource(graph!.getOutput("G.diffuseOpacity")!);
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

for (const [name, pass, props] of [
    ["rayCones", "GBufferRT", ", 'texLOD': 'RayCones'"],
    ["rayDiffs", "GBufferRT", ", 'texLOD': 'RayDiffs'"],
    ["implicit", "GBufferRaster", ""],
] as const) {
    gpuTest(`TextureArrayLOD.${name}IndependentOfOtherTextures`, async ({ device }) => {
        await initScripting("/node_modules/pyodide");
        const alone = await render(device, pass, props, false);
        const shared = await render(device, pass, props, true);
        let worst = 0;
        let covered = 0;
        let varied = 0;
        for (let i = 0; i < alone.length; i += 4) {
            if (alone[i + 3]! > 0) covered++;
            if (Math.abs(alone[i]! - alone[4]!) > 1e-3) varied++;
            for (let c = 0; c < 3; c++) worst = Math.max(worst, Math.abs(alone[i + c]! - shared[i + c]!));
        }
        console.error(`# texture-array-lod ${name}: covered=${covered} varied=${varied} worst=${worst.toExponential(2)}`);
        expectEq(covered > size * size * 0.3 && varied > 1000, true, "the floor covers the view with texture detail");
        expectEq(worst <= 1e-5, true, `rendering changed when a larger texture joined the array (worst ${worst})`);
    });
}
