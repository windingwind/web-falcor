/**
 * Animated emissive geometry: Scene.animate() rebuilds the LightCollection when an
 * emissive instance's matrix changes (native LightCollection::update MatrixChanged) and
 * bumps emissiveVersion; LightBVHSampler.refit() then updates the tree in place
 * (native LightBVH::refit) and agrees with a fresh build over the moved triangles.
 */

import { LightBVHSampler, Scene, float2, float3, float4, quatf, kDefaultLightBVHOptions, kDefaultLightBVHSamplerOptions } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq, expectClose } from "../harness/registry.js";

gpuTest("AnimatedEmissive.lightCollectionFollowsAnimationAndBvhRefits", async ({ device }) => {
    const quad = (y: number) => [
        { position: new float3(-0.5, y - 0.5, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(0.5, y - 0.5, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0.5, y + 0.5, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 1) },
        { position: new float3(-0.5, y + 0.5, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    // Second quad lies in the XZ plane at y=3 (normal -y): distinct normals keep the
    // lighting cones valid — native's builder invalidates a leaf whose triangles all
    // share one normal (sinTotalTheta == 0), while its refit kernel does not.
    const tiltedQuad = [
        { position: new float3(-0.5, 3, -0.5), normal: new float3(0, -1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(0.5, 3, -0.5), normal: new float3(0, -1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0.5, 3, 0.5), normal: new float3(0, -1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 1) },
        { position: new float3(-0.5, 3, 0.5), normal: new float3(0, -1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
    // Node 0 slides the emissive quad +2 in x over one second; node 1 (identity, unanimated) holds a static one.
    const nodes = [
        { parent: -1, t: new float3(0, 0, 0), r: quatf.identity(), s: new float3(1, 1, 1) },
        { parent: -1, t: new float3(0, 0, 0), r: quatf.identity(), s: new float3(1, 1, 1) },
    ];
    const animations = [{ nodeID: 0, path: "translation" as const, times: new Float32Array([0, 1]), values: new Float32Array([0, 0, 0, 2, 0, 0]), interp: "LINEAR" as const }];
    const scene = new Scene(
        device,
        [
            { vertices: quad(0), indices, materialID: 0, nodeID: 0 },
            { vertices: tiltedQuad, indices, materialID: 0, nodeID: 1 },
        ],
        [{ basic: { baseColor: new float4(1, 1, 1, 1), emissive: new float3(5, 5, 5) }, header: { emissive: true } }],
        [],
        undefined,
        [],
        nodes,
        animations,
    );
    expectEq(scene.isAnimated(), true, "scene animates");
    const tris0 = scene.getEmissiveTriangles();
    expectEq(tris0.length, 4, "two emissive quads = 4 triangles");
    const v0 = scene.emissiveVersion;
    // One triangle per leaf: root + internal nodes exercise the cone-union refit path.
    const options = { ...kDefaultLightBVHSamplerOptions, buildOptions: { ...kDefaultLightBVHOptions, maxTriangleCountPerLeaf: 1 } };
    const sampler = new LightBVHSampler(device, tris0, options);
    expectEq(sampler.valid, true, "BVH built");
    const before = Array.from({ length: sampler.getNodeCount() }, (_v, i) => sampler.getNodeAttributes(i));

    // Static frame (t = 0 keeps the identity matrix): no LightCollection rebuild.
    scene.animate(0);
    expectEq(scene.emissiveVersion, v0, "no emissive change at t=0");

    scene.animate(0.5);
    expectEq(scene.emissiveVersion, v0 + 1, "emissive version bumped by the moved instance");
    const tris1 = scene.getEmissiveTriangles();
    expectClose(tris1[0]!.posW[0]![0], tris0[0]!.posW[0]![0] + 1, 1e-5, "quad 0 moved +1 in x at t=0.5");
    expectClose(tris1[2]!.posW[0]![0], tris0[2]!.posW[0]![0], 1e-6, "quad 1 static");
    expectClose(tris1[0]!.flux, tris0[0]!.flux, 1e-6, "flux unchanged by translation");

    // Repeating the same time re-evaluates identical matrices: no rebuild.
    scene.animate(0.5);
    expectEq(scene.emissiveVersion, v0 + 1, "identical matrices do not rebuild");

    // Refit keeps the structure and reproduces a fresh build's bounds/cones.
    expectEq(sampler.refit(tris1), true, "refit accepted");
    const fresh = new LightBVHSampler(device, tris1, options);
    expectEq(fresh.getNodeCount(), sampler.getNodeCount(), "same node count");
    expectEq(fresh.getNodeCount() >= 5, true, `multi-level tree (${fresh.getNodeCount()} nodes)`);
    let maxOriginDiff = 0;
    for (let i = 0; i < fresh.getNodeCount(); i++) {
        const a = sampler.getNodeAttributes(i);
        const b = fresh.getNodeAttributes(i);
        const fmt = (n: typeof a) => `o=${n.origin.map((v) => v.toFixed(3))} e=${n.extent.map((v) => v.toFixed(3))} dir=${n.coneDirection.map((v) => v.toFixed(3))} cos=${n.cosConeAngle.toFixed(4)} flux=${n.flux.toFixed(3)}`;
        if (i < 3) console.error(`#  node ${i}: before ${fmt(before[i]!)}\n#          refit  ${fmt(a)}\n#          fresh  ${fmt(b)}`);
        for (let k = 0; k < 3; k++) {
            maxOriginDiff = Math.max(maxOriginDiff, Math.abs(a.origin[k]! - b.origin[k]!), Math.abs(a.extent[k]! - b.extent[k]!));
            // Cones compare only where both paths keep them valid: the native builder invalidates
            // single-triangle / identical-normal leaves (sinTotalTheta == 0) whereas the native refit
            // kernel yields the tight cone (cos = 1) — mirrored here, invalid cones are ignored anyway.
            if (a.cosConeAngle > -1 && b.cosConeAngle > -1) expectClose(a.coneDirection[k]!, b.coneDirection[k]!, 1e-3, `node ${i} cone dir`);
        }
        if (a.cosConeAngle > -1 && b.cosConeAngle > -1) expectClose(a.cosConeAngle, b.cosConeAngle, 1e-3, `node ${i} cos cone`);
        expectEq(a.cosConeAngle >= -1 && a.cosConeAngle <= 1, true, `node ${i} refit cone in range`);
        expectClose(a.flux, b.flux, 1e-6, `node ${i} flux`);
    }
    console.error(`# refit vs rebuild: nodes=${fresh.getNodeCount()} maxOriginDiff=${maxOriginDiff.toExponential(2)}`);
    expectEq(maxOriginDiff < 1e-4, true, `refit bounds match rebuild (${maxOriginDiff})`);
    // Root moved: the union of a quad at x in [0.5,1.5] and one at [-0.5,0.5] is centered at x=0.5.
    expectClose(sampler.getNodeAttributes(0).origin[0]!, 0.5, 1e-4, "root origin x after move");
    expectClose(before[0]!.origin[0]!, 0, 1e-4, "root origin x before move");

    // A different triangle count rejects the refit (caller rebuilds).
    expectEq(sampler.refit(tris1.slice(0, 2)), false, "size change rejects refit");
});
