/**
 * Compressed SparseBrickSet bricks (createSBS(compressed=True)). The port of BC4Encode.slang's
 * compressBlock feeds a real BC4Snorm brick texture, as natively. Checks: the packed blocks decode
 * on the GPU to the BC4 palette within the hardware's interpolation precision (this GPU does not
 * use exact 1/7 steps, which is why the texture stays BC4 rather than CPU-decoded floats); a
 * compressed SBS scene renders the same surface as the uncompressed one up to quantization.
 * Native can't render an SBS on this machine for an image oracle (feature-sbs.gpu.test.ts).
 */

import { Mt19937, ResourceFormat, SDFSBS, compressBC4Block, decodeBC4Block, initScripting, packBC4Block, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";
import { GPUUnitTestContext } from "../harness/unit-test-context.js";

const kShader = `
Texture2D<float> tex;
RWStructuredBuffer<float> result;
cbuffer CB { uint width; uint count; }
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= count) return;
    result[tid.x] = tex.Load(int3(tid.x % width, tid.x / width, 0));
}
`;

gpuTest("SBS.bc4Compressed", async ({ device }) => {
    const rng = new Mt19937(7);
    const blocks: Int32Array[] = [new Int32Array(16).fill(0), new Int32Array(16).fill(127), new Int32Array(16).fill(-128), Int32Array.from({ length: 16 }, (_, i) => i * 16 - 128)];
    for (let b = 0; b < 60; b++) blocks.push(Int32Array.from({ length: 16 }, () => Math.floor((rng.next() / 2 ** 32) * 256) - 128));
    const n = blocks.length;
    const width = 4 * n;
    const data = new Uint8Array(8 * n);
    const expected = new Float32Array(width * 4);
    let quantError = 0;
    blocks.forEach((block, b) => {
        const { alpha0, alpha1, indices } = compressBC4Block(block);
        data.set(packBC4Block(alpha0, alpha1, indices), 8 * b);
        const decoded = decodeBC4Block(alpha0, alpha1, indices);
        for (let i = 0; i < 16; i++) {
            expected[(i >> 2) * width + 4 * b + (i & 3)] = decoded[i]!;
            quantError = Math.max(quantError, Math.abs(decoded[i]! - Math.max(block[i]! / 127, -1)));
        }
    });
    const tex = device.createTexture2D(width, 4, ResourceFormat.BC4Snorm, 1, 1, data);
    const ctx = new GPUUnitTestContext(device);
    ctx.createProgramFromModules([{ sources: [{ string: kShader }] }]);
    ctx.vars()["tex"] = tex;
    ctx.vars()["CB"]["width"] = width;
    ctx.vars()["CB"]["count"] = width * 4;
    ctx.allocateStructuredBuffer("result", width * 4);
    ctx.runProgram(width * 4, 1, 1);
    const gpu = await ctx.readBuffer("result", Float32Array);
    let worst = 0;
    for (let i = 0; i < width * 4; i++) worst = Math.max(worst, Math.abs(gpu[i]! - expected[i]!));
    console.error(`# bc4: ${n} blocks, encoder error ${quantError.toFixed(3)}, GPU vs exact palette ${worst.toExponential(2)}`);
    // This GPU interpolates BC4 SNORM coarsely (up to ~0.053 off the exact palette next to a -128 endpoint).
    expectEq(worst < 0.06, true, `the blocks decode to their palette (worst ${worst})`);
    expectEq(quantError < 0.35, true, `encoder error ${quantError}`);

    // createSBS(compressed=True) validates the brick width like native.
    let threw = false;
    try {
        new SDFSBS(6, true);
    } catch {
        threw = true;
    }
    expectEq(threw, true, "brickWidth must be 4k-1 when compressed");

    // SDFSBS.pyscene compressed vs uncompressed: SceneDebugger GeometryID footprints.
    await initScripting("/node_modules/pyodide");
    const base = "/Falcor/tests/image_tests/scene/scenes";
    const source = await (await fetch(`${base}/SDFSBS.pyscene`)).text();
    const footprint = async (src: string) => {
        const [graph] = await runGraphScript(device, "from falcor import *\ng = RenderGraph('D')\ng.addPass(createPass('SceneDebugger', {'mode': 'GeometryID'}), 'D')\ng.markOutput('D.output')\nm.addGraph(g)\n");
        const scene = await runSceneScript(device, src, base);
        scene.camera.setAspectRatio(1);
        graph!.onResize(256, 256);
        graph!.setScene(scene);
        graph!.execute(device.renderContext);
        const out = new Float32Array((await device.renderContext.readTextureSubresource(graph!.getOutput("D.output")!)).buffer);
        return Uint8Array.from({ length: 256 * 256 }, (_, i) => (Math.abs(out[i * 4]! - 0.153) < 0.05 ? 1 : 0));
    };
    const plain = await footprint(source);
    const compressed = await footprint(source.replace("SDFGrid.createSBS()", "SDFGrid.createSBS(compressed=True)"));
    const hits = plain.reduce((a, b) => a + b, 0);
    const differ = plain.reduce((a, v, i) => a + (v !== compressed[i] ? 1 : 0), 0);
    console.error(`# bc4 SBS: ${hits} body pixels, ${differ} differ when compressed`);
    expectEq(hits > 5000, true, `the SBS renders (${hits})`);
    expectEq(differ < hits * 0.05, true, `compressed footprint (${differ} of ${hits} differ)`);
});
