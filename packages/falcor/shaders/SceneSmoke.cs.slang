/**
 * gScene smoke test: reads camera, vertex, instance and material data through
 * the unmodified upstream Scene module (via WebFalcor overrides).
 */
import Scene.Scene;

RWStructuredBuffer<float4> gOut;

[numthreads(16, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= 8) return;

    switch (tid.x)
    {
    case 0:
    {
        // Camera viewProj row 0.
        float4x4 m = gScene.camera.data.viewProjMat;
        gOut[0] = float4(m[0][0], m[0][1], m[0][2], m[0][3]);
        break;
    }
    case 1:
    {
        // First vertex position (world = local, identity transform).
        StaticVertexData v = gScene.getVertex(0);
        gOut[1] = float4(v.position, 1.0);
        break;
    }
    case 2:
    {
        // Second vertex normal (unpacked through PackedStaticVertexData).
        StaticVertexData v = gScene.getVertex(1);
        gOut[2] = float4(v.normal, 0.0);
        break;
    }
    case 3:
    {
        // Instance data.
        GeometryInstanceID iid;
        iid.index = 0;
        GeometryInstanceData inst = gScene.getGeometryInstance(iid);
        gOut[3] = float4(inst.materialID, inst.vbOffset, inst.ibOffset, 1.0);
        break;
    }
    case 4:
    {
        // Triangle indices via mesh/index buffer.
        uint3 idx = gScene.getIndices(0, 0);
        gOut[4] = float4(idx.x, idx.y, idx.z, 0.0);
        break;
    }
    case 5:
    {
        // Material: base color through the material system (uniform-mode handle).
        BasicMaterialData md = gScene.materials.getBasicMaterialData(0);
        gOut[5] = float4(md.baseColor);
        break;
    }
    case 6:
    {
        // Material header: type + IoR.
        MaterialHeader header = gScene.materials.getMaterialHeader(0);
        gOut[6] = float4(float(header.getMaterialType()), header.getIoR(), header.isBasicMaterial() ? 1.0 : 0.0, 0.0);
        break;
    }
    case 7:
    {
        // World-space transform of vertex 0 through the instance transform chain.
        GeometryInstanceID iid2;
        iid2.index = 0;
        float4x4 worldMat = gScene.getWorldMatrix(iid2);
        StaticVertexData v = gScene.getVertex(0);
        gOut[7] = mul(worldMat, float4(v.position, 1.0));
        break;
    }
    }
}
