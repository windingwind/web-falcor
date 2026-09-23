/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * WebFalcor override of Samples/HelloDXR/HelloDXR.3d.slang: WGSL has no SV_PrimitiveID,
 * so the scene is drawn with vertex pulling (non-indexed draws, triangleIndex =
 * SV_VertexID / 3, as the GBufferRaster override) and psMain gets the triangle index as a
 * flat varying. DispatchRaysIndex() in native's pixel shader becomes the pixel position,
 * and the implicit-LOD sampler becomes explicit gradients taken up front (WGSL rejects
 * implicit derivatives in the material code's non-uniform control flow; same LODs).
 */
import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;

struct PullVSIn
{
    uint vertexID : SV_VertexID;
    uint instanceID : DRAW_ID;
};

struct PullVSOut
{
    float3 normalW : NORMAL;
    float4 tangentW : TANGENT;
    float2 texC : TEXCRD;
    float3 posW : POSW;
    nointerpolation uint instanceIdx : INSTANCE_IDX;
    nointerpolation uint materialID : MATERIAL_ID;
    nointerpolation uint triangleIndex : TRI_INDEX;
    float4 posH : SV_POSITION;
};

/** defaultVS, with the vertex pulled from the scene buffers. */
PullVSOut vsMain(PullVSIn vIn)
{
    PullVSOut vOut;
    GeometryInstanceID instanceID;
    instanceID.index = vIn.instanceID;
    const uint triangleIndex = vIn.vertexID / 3;
    const uint corner = vIn.vertexID % 3;
    const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    const uint3 localIndices = gScene.getLocalIndices(instance.ibOffset, triangleIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));
    const StaticVertexData v = gScene.getVertex(instance.vbOffset + localIndices[corner]);

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    vOut.posW = mul(worldMat, float4(v.position, 1.f)).xyz;
    vOut.posH = mul(gScene.camera.getViewProj(), float4(vOut.posW, 1.f));
    vOut.instanceIdx = vIn.instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);
    vOut.texC = v.texCrd;
    vOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), v.normal);
    vOut.tangentW = float4(mul((float3x3)worldMat, v.tangent.xyz), v.tangent.w);
    vOut.triangleIndex = triangleIndex;
    return vOut;
}

/** Scene.Raster's prepareVertexData/alphaTest/prepareShadingData for the pulled varyings. */
VertexData pulledVertexData(PullVSOut vsOut)
{
    GeometryInstanceID instanceID;
    instanceID.index = vsOut.instanceIdx;
    VertexData v;
    v.posW = vsOut.posW;
    v.texC = vsOut.texC;
    v.normalW = normalize(vsOut.normalW);
    v.faceNormalW = gScene.getFaceNormalW(instanceID, vsOut.triangleIndex);
    v.tangentW.xyz = normalize(vsOut.tangentW.xyz);
    v.tangentW.w = sign(vsOut.tangentW.w); // Preserve zero to indicate invalid tangent.
    return v;
}

float4 psMain(PullVSOut vsOut) : SV_TARGET
{
    let lod = ExplicitGradientTextureSampler(ddx(vsOut.texC), ddy(vsOut.texC));
    VertexData v = pulledVertexData(vsOut);
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
    float3 viewDir = normalize(gScene.camera.getPosition() - vsOut.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir);

    // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    float3 color = mi.getProperties(sd).emission;

    uint2 launchIndex = uint2(vsOut.posH.xy);
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(launchIndex, 0);

    // Direct lighting from analytic light sources
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, gScene.getLight(i), ls);
        color += mi.eval(sd, ls.dir, sg) * ls.Li;
    }

    return float4(color, 1.f);
}
