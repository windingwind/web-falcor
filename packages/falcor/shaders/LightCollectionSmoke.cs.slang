/**
 * WebFalcor GPU test kernel: reads gScene.lightCollection back so the host
 * packing (PackedEmissiveTriangle 64B, EmissiveFlux WGSL layout, MeshLightData,
 * active lists) can be verified against CPU expectations.
 */

import Scene.Scene;

RWStructuredBuffer<float4> gOut;

[numthreads(1, 1, 1)]
void main()
{
    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(1);
    gOut[0] = float4(tri.posW[0], float(gScene.lightCollection.getTriangleCount()));
    gOut[1] = float4(tri.posW[1], float(gScene.lightCollection.getActiveTriangleCount()));
    gOut[2] = float4(tri.posW[2], tri.area);
    gOut[3] = float4(tri.normal, float(tri.materialID));
    gOut[4] = float4(gScene.lightCollection.getAverageRadiance(1), gScene.lightCollection.fluxData[1].flux);
    gOut[5] = float4(tri.texCoords[0], tri.texCoords[2]);
    gOut[6] = float4(
        float(gScene.lightCollection.meshData[0].instanceID),
        float(gScene.lightCollection.meshData[0].triangleOffset),
        float(gScene.lightCollection.meshData[0].triangleCount),
        float(gScene.lightCollection.meshData[0].materialID));
    gOut[7] = float4(
        asfloat(gScene.lightCollection.perMeshInstanceOffset[0]),
        float(gScene.lightCollection.perMeshInstanceOffset[1]),
        float(gScene.lightCollection.activeTriangles[1]),
        float(gScene.lightCollection.triToActiveMapping[1]));
}
