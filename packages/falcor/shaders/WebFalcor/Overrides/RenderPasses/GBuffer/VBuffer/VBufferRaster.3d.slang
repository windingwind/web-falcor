/**
 * WebFalcor override of RenderPasses/GBuffer/VBuffer/VBufferRaster.3d.slang
 * (docs §9). WGSL fragment shaders have no SV_Barycentrics/SV_PrimitiveID,
 * so the pass draws NON-INDEXED (3 * triangleCount vertices) and pulls
 * vertices from the scene buffers: barycentrics interpolate from per-corner
 * unit weights and the triangle index rides a flat varying. WebGPU has no
 * rasterizer-ordered views, so the optional motion-vector and mask channels
 * are extra render targets (SV_TARGET1/2) instead of UAV stores.
 */
import Scene.Raster;
import Utils.Math.MathHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

#define is_valid(name) (is_valid_##name != 0)

struct VBufferPSOut
{
    PackedHitInfo packedHitInfo : SV_TARGET0;
#if is_valid(gMotionVector)
    float2 mvec : SV_TARGET1;
#endif
#if is_valid(gMask)
    float mask : SV_TARGET2;
#endif
};

struct VBufferVSIn
{
    uint vertexID : SV_VertexID;
    uint instanceID : DRAW_ID;
};

struct VBufferVSOut
{
    float2 texC : TEXCRD;
    float3 bary : BARYWEIGHTS; ///< Rasterizer-interpolated barycentrics.

    // Per-triangle data
    nointerpolation uint instanceIdx : INSTANCE_IDX;
    nointerpolation uint triangleIndex : TRI_INDEX;
    nointerpolation uint materialID : MATERIAL_ID;

    float4 posH : SV_POSITION;

#if is_valid(gMotionVector)
    float4 prevPosH : PREVPOSH; ///< Position in clip space for the previous frame.
#endif
};

VBufferVSOut vsMain(VBufferVSIn vsIn)
{
    VBufferVSOut vsOut;
    GeometryInstanceID instanceID;
    instanceID.index = vsIn.instanceID;

    const uint triangleIndex = vsIn.vertexID / 3;
    const uint corner = vsIn.vertexID % 3;
    const uint3 vtxIndices = gScene.getIndices(instanceID, triangleIndex);
    const StaticVertexData v = gScene.getVertex(vtxIndices[corner]);

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(v.position, 1.f)).xyz;
    vsOut.posH = mul(gScene.camera.getViewProj(), float4(posW, 1.f));

    vsOut.texC = v.texCrd;
    vsOut.bary = float3(corner == 0 ? 1.f : 0.f, corner == 1 ? 1.f : 0.f, corner == 2 ? 1.f : 0.f);
    vsOut.instanceIdx = vsIn.instanceID;
    vsOut.triangleIndex = triangleIndex;
    vsOut.materialID = gScene.getMaterialID(instanceID);

#if is_valid(gMotionVector)
    // Compute the vertex position in the previous frame (dynamic meshes keep
    // their previous positions in prevVertices at prevVbOffset + local index).
    float3 prevPos = v.position;
    GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    if (instance.isDynamic())
    {
        const uint3 localIndices = gScene.getLocalIndices(instance.ibOffset, triangleIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));
        prevPos = gScene.prevVertices[gScene.meshes[instance.geometryID].prevVbOffset + localIndices[corner]].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vsOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
#endif
    return vsOut;
}

VBufferPSOut psMain(VBufferVSOut vsOut)
{
    VBufferPSOut psOut;

#if USE_ALPHA_TEST
    // Alpha textures sampled at lod=0 to match the ray-traced V-buffer.
    VertexData v = {};
    v.texC = vsOut.texC;
    if (gScene.materials.alphaTest(v, vsOut.materialID, 0.f))
        discard;
#endif

    // DXR stores barycentric weights for vertices B and C; same here.
    TriangleHit triangleHit;
    triangleHit.instanceID.index = vsOut.instanceIdx;
    triangleHit.primitiveIndex = vsOut.triangleIndex;
    triangleHit.barycentrics = vsOut.bary.yz;
    psOut.packedHitInfo = triangleHit.pack();

#if is_valid(gMask)
    psOut.mask = 1.f;
#endif

#if is_valid(gMotionVector)
    {
        const int2 ipos = int2(vsOut.posH.xy);
        // Current sample in pixel coords; previous-frame sample in unjittered clip space.
        const float2 pixelPos = ipos + float2(0.5, 0.5);
        // Remove camera jitter from motion vector
        psOut.mvec = calcMotionVector(pixelPos, vsOut.prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
    }
#endif
    return psOut;
}
