/**
 * M5 glTF import GPU test: an embedded glTF scene (quad, node transform,
 * pbrMetallicRoughness material) imports and renders through GBufferRaster.
 */

import { GltfImporter, RenderGraph, createPass, float3 } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectClose, expectEq } from "../harness/registry.js";

function makeQuadGltf(): Uint8Array {
    // Unit quad in the xy plane (two triangles), translated to z = -1 by its node.
    const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
    const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
    const uvs = new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]);
    const indices = new Uint16Array([0, 1, 2, 0, 2, 3]);

    const bin = new Uint8Array(positions.byteLength + normals.byteLength + uvs.byteLength + indices.byteLength);
    let off = 0;
    for (const arr of [positions, normals, uvs, indices]) {
        bin.set(new Uint8Array(arr.buffer), off);
        off += arr.byteLength;
    }
    const b64 = btoa(String.fromCharCode(...bin));

    const gltf = {
        asset: { version: "2.0" },
        scene: 0,
        scenes: [{ nodes: [0] }],
        nodes: [{ mesh: 0, translation: [0, 0, -1] }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1, TEXCOORD_0: 2 }, indices: 3, material: 0 }] }],
        materials: [{ pbrMetallicRoughness: { baseColorFactor: [0.2, 0.4, 0.8, 1.0], roughnessFactor: 0.5, metallicFactor: 0.0 } }],
        accessors: [
            { bufferView: 0, componentType: 5126, count: 4, type: "VEC3" },
            { bufferView: 1, componentType: 5126, count: 4, type: "VEC3" },
            { bufferView: 2, componentType: 5126, count: 4, type: "VEC2" },
            { bufferView: 3, componentType: 5123, count: 6, type: "SCALAR" },
        ],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: positions.byteLength },
            { buffer: 0, byteOffset: positions.byteLength, byteLength: normals.byteLength },
            { buffer: 0, byteOffset: positions.byteLength + normals.byteLength, byteLength: uvs.byteLength },
            { buffer: 0, byteOffset: positions.byteLength + normals.byteLength + uvs.byteLength, byteLength: indices.byteLength },
        ],
        buffers: [{ uri: `data:application/octet-stream;base64,${b64}`, byteLength: bin.byteLength }],
    };
    return new TextEncoder().encode(JSON.stringify(gltf));
}

gpuTest("GltfImporter.quadRendersThroughGBuffer", async ({ device }) => {
    const scene = await GltfImporter.importFromBytes(device, makeQuadGltf());
    scene.camera.setPosition(new float3(0.5, 0.5, 2));
    scene.camera.setTarget(new float3(0.5, 0.5, -1));
    scene.camera.setAspectRatio(1);

    const size = 32;
    const graph = new RenderGraph(device, "GltfGraph");
    graph.onResize(size, size);
    graph.addPass(createPass(device, "GBufferRaster"), "GBufferRaster");
    graph.markOutput("GBufferRaster.posW");
    graph.markOutput("GBufferRaster.texC");
    graph.setScene(scene);

    const ctx = device.renderContext;
    graph.execute(ctx);

    const center = (size / 2) * size + size / 2;
    const posW = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.posW")!)).buffer);
    // Node translation moved the quad to z = -1; camera center ray hits (0.5, 0.5, -1).
    expectClose(posW[center * 4 + 0]!, 0.5, 0.08, "posW.x (half-pixel offset)");
    expectClose(posW[center * 4 + 1]!, 0.5, 0.08, "posW.y (half-pixel offset)");
    expectClose(posW[center * 4 + 2]!, -1.0, 1e-3, "posW.z (node transform applied)");

    const texC = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("GBufferRaster.texC")!)).buffer);
    expectClose(texC[center * 2]!, 0.5, 0.08, "texC.u");
    expectClose(texC[center * 2 + 1]!, 0.5, 0.08, "texC.v");

    expectEq(scene.getGeometryInstanceCount(), 1, "one instance");
});
