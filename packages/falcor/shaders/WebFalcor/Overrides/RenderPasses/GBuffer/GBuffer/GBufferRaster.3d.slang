/**
 * WebFalcor override of RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang
 * (docs §4.3/§9). Diffs vs upstream: psMain drops SV_PrimitiveID and
 * SV_Barycentrics (absent from WGSL) and [earlydepthstencil]; the face normal
 * comes from screen-space derivatives. WebGPU has no rasterizer-ordered
 * views, so the "extra" UAV channels are render targets too: the host assigns
 * every connected channel a batch + location (LOC_<texname>, is_valid_<texname>)
 * and draws one geometry pass per batch of <= 8 targets. The vbuffer channel
 * needs the primitive id: with WEBFALCOR_VERTEX_PULLING the host draws
 * non-indexed and the VS pulls vertices itself (triangle index + per-corner
 * barycentrics ride flat/interpolated varyings, like VBufferRaster).
 */
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
#include "Scene/VertexAttrib.slangh"

import Scene.Raster;
import Utils.Math.MathHelpers;
import GBufferHelpers;
import Rendering.Materials.TexLODHelpers;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
};

#define is_valid(name) (is_valid_##name != 0)

// Render-target location per channel, assigned by the host for the current batch.
#define WEBFALCOR_CAT_(a, b) a##b
#define WEBFALCOR_CAT(a, b) WEBFALCOR_CAT_(a, b)
#define WEBFALCOR_TARGET(name) WEBFALCOR_CAT(SV_TARGET, LOC_##name)

// GBuffer channels present in this batch (members exist only for attached targets).
struct GBufferPSOut
{
#if is_valid(gPosW)
    float4 posW : WEBFALCOR_TARGET(gPosW);
#endif
#if is_valid(gNormW)
    float4 normW : WEBFALCOR_TARGET(gNormW);
#endif
#if is_valid(gTangentW)
    float4 tangentW : WEBFALCOR_TARGET(gTangentW);
#endif
#if is_valid(gFaceNormalW)
    float4 faceNormalW : WEBFALCOR_TARGET(gFaceNormalW);
#endif
#if is_valid(gTexC)
    float2 texC : WEBFALCOR_TARGET(gTexC);
#endif
#if is_valid(gTexGrads)
    float4 texGrads : WEBFALCOR_TARGET(gTexGrads);
#endif
#if is_valid(gMotionVector)
    float2 mvec : WEBFALCOR_TARGET(gMotionVector);
#endif
#if is_valid(gMaterialData)
    uint4 mtlData : WEBFALCOR_TARGET(gMaterialData);
#endif
#if is_valid(gGuideNormalW)
    float4 guideNormalW : WEBFALCOR_TARGET(gGuideNormalW);
#endif
#if is_valid(gDiffOpacity)
    float4 diffuseOpacity : WEBFALCOR_TARGET(gDiffOpacity);
#endif
#if is_valid(gSpecRough)
    float4 specRough : WEBFALCOR_TARGET(gSpecRough);
#endif
#if is_valid(gEmissive)
    float4 emissive : WEBFALCOR_TARGET(gEmissive);
#endif
#if is_valid(gViewW)
    float4 viewW : WEBFALCOR_TARGET(gViewW);
#endif
#if is_valid(gPosNormalFwidth)
    float2 pnFwidth : WEBFALCOR_TARGET(gPosNormalFwidth);
#endif
#if is_valid(gLinearZAndDeriv)
    float2 linearZ : WEBFALCOR_TARGET(gLinearZAndDeriv);
#endif
#if is_valid(gMask)
    float mask : WEBFALCOR_TARGET(gMask);
#endif
#if is_valid(gVBuffer)
    PackedHitInfo vbuffer : WEBFALCOR_TARGET(gVBuffer);
#endif
};

#ifndef WEBFALCOR_VERTEX_PULLING
#define WEBFALCOR_VERTEX_PULLING 0
#endif

#if WEBFALCOR_VERTEX_PULLING
struct PullVSIn
{
    uint vertexID : SV_VertexID;
    uint instanceID : DRAW_ID;
};

/// VSOut plus the primitive data the pulled path can provide (WGSL has no SV_PrimitiveID/SV_Barycentrics).
struct PullVSOut
{
    float3 normalW : NORMAL;
    float4 tangentW : TANGENT;
    float2 texC : TEXCRD;
    float3 posW : POSW;
    float4 prevPosH : PREVPOSH;
    float3 bary : BARYWEIGHTS;
    nointerpolation uint instanceIdx : INSTANCE_IDX;
    nointerpolation uint materialID : MATERIAL_ID;
    nointerpolation uint triangleIndex : TRI_INDEX;
    float4 posH : SV_POSITION;
};

PullVSOut vsMain(PullVSIn vsIn)
{
    PullVSOut vsOut;
    GeometryInstanceID instanceID;
    instanceID.index = vsIn.instanceID;

    const uint triangleIndex = vsIn.vertexID / 3;
    const uint corner = vsIn.vertexID % 3;
    const GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    const uint3 localIndices = gScene.getLocalIndices(instance.ibOffset, triangleIndex, instance.flags & uint(GeometryInstanceFlags::Use16BitIndices));
    const StaticVertexData v = gScene.getVertex(instance.vbOffset + localIndices[corner]);

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(v.position, 1.f)).xyz;
    vsOut.posW = posW;
    vsOut.posH = mul(gScene.camera.getViewProj(), float4(posW, 1.f));
    vsOut.instanceIdx = vsIn.instanceID;
    vsOut.materialID = gScene.getMaterialID(instanceID);
    vsOut.texC = v.texCrd;
    vsOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), v.normal);
    vsOut.tangentW = float4(mul((float3x3)worldMat, v.tangent.xyz), v.tangent.w);
    vsOut.bary = float3(corner == 0 ? 1.f : 0.f, corner == 1 ? 1.f : 0.f, corner == 2 ? 1.f : 0.f);
    vsOut.triangleIndex = triangleIndex;

    // Previous-frame position (dynamic meshes keep prev positions at prevVbOffset + local index).
    float3 prevPos = v.position;
    if (instance.isDynamic())
    {
        prevPos = gScene.prevVertices[gScene.meshes[instance.geometryID].prevVbOffset + localIndices[corner]].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vsOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));
    return vsOut;
}

#else
VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}
#endif

float2 computeMotionVector(const float4 prevPosH, const int2 ipos)
{
    // Current sample in pixel coords.
    float2 pixelPos = ipos + float2(0.5, 0.5);
    // Sample in previous frame in clip space coords, no jittering applied.
    // Remove camera jitter from motion vector
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
}

/** Shared G-buffer shading for both vertex paths (VSOut carries member semantics, so the
    pulled path hands over plain vertex data instead of rebuilding a VSOut). */
GBufferPSOut shadeGBuffer(const VertexData v, const uint materialID, const float4 posH, const float4 prevPosH,
    const GeometryInstanceID instanceID, const uint triangleIndex, const float3 barycentrics, const float2 texCdx, const float2 texCdy)
{
    // Using posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(posH.xy);

    // WebFalcor: WGSL requires uniform control flow for implicit-LOD sampling;
    // uv derivatives are taken by the caller and sampling uses explicit gradients (equivalent).
    let lod = ExplicitGradientTextureSampler(texCdx, texCdy);

#if USE_ALPHA_TEST
    if (gScene.materials.alphaTest(v, materialID, lod))
        discard;
#endif
    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, viewDir);

    uint hints = 0;
#if ADJUST_SHADING_NORMALS
    hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#endif

    // Create material instance and query its properties.
    let mi = gScene.materials.getMaterialInstance(sd, lod, hints);
    let bsdfProperties = mi.getProperties(sd);

    const GBufferData gbuf = prepareGBufferData(sd, v, mi, bsdfProperties);

    GBufferPSOut psOut = {};

    // Store render target outputs.
#if is_valid(gPosW)
    psOut.posW = gbuf.posW;
#endif
#if is_valid(gNormW)
    psOut.normW = gbuf.normW;
#endif
#if is_valid(gTangentW)
    psOut.tangentW = gbuf.tangentW;
#endif
#if is_valid(gFaceNormalW)
    psOut.faceNormalW = gbuf.faceNormalW;
#endif
#if is_valid(gTexC)
    psOut.texC = gbuf.texC;
#endif
#if is_valid(gTexGrads)
    psOut.texGrads = float4(ddx(sd.uv), ddy(sd.uv));
#endif
#if is_valid(gMotionVector)
    psOut.mvec = computeMotionVector(prevPosH, ipos);
#endif
#if is_valid(gMaterialData)
    psOut.mtlData = gbuf.mtlData;
#endif

    // Upstream UAV outputs, written as render targets here.
#if is_valid(gViewW)
    psOut.viewW = float4(sd.V, 0.f);
#endif
#if is_valid(gGuideNormalW)
    psOut.guideNormalW = gbuf.guideNormalW;
#endif
#if is_valid(gDiffOpacity)
    psOut.diffuseOpacity = gbuf.diffuseOpacity;
#endif
#if is_valid(gSpecRough)
    psOut.specRough = gbuf.specRough;
#endif
#if is_valid(gEmissive)
    psOut.emissive = gbuf.emissive;
#endif
#if is_valid(gMask)
    psOut.mask = 1.f;
#endif

    // Length of derivatives of position and guide normal.
    // This is a feature guide for the SVGF denoiser pass.
#if is_valid(gPosNormalFwidth)
    psOut.pnFwidth = float2(length(fwidth(sd.posW)), length(fwidth(bsdfProperties.guideNormal)));
#endif

    // Linear z and its derivative
#if is_valid(gLinearZAndDeriv)
    {
        const float linearZ = posH.z * posH.w;
        psOut.linearZ = float2(linearZ, max(abs(ddx(linearZ)), abs(ddy(linearZ))));
    }
#endif

    // Store hit information (DXR stores barycentric weights for vertices B and C).
#if is_valid(gVBuffer) && WEBFALCOR_VERTEX_PULLING
    {
        TriangleHit triangleHit;
        triangleHit.instanceID = instanceID;
        triangleHit.primitiveIndex = triangleIndex;
        triangleHit.barycentrics = barycentrics.yz;
        psOut.vbuffer = triangleHit.pack();
    }
#endif

    return psOut;
}

// WebFalcor: SV_PrimitiveID/SV_Barycentrics are unavailable on the WGSL target
// ([earlydepthstencil] likewise). Without vertex pulling the face normal is
// derived from position derivatives -- equivalent for planar triangles.
#if WEBFALCOR_VERTEX_PULLING
GBufferPSOut psMain(PullVSOut pIn)
{
    GeometryInstanceID instanceID;
    instanceID.index = pIn.instanceIdx;
    VertexData v;
    v.posW = pIn.posW;
    v.texC = pIn.texC;
    v.normalW = normalize(pIn.normalW);
    v.faceNormalW = gScene.getFaceNormalW(instanceID, pIn.triangleIndex);
    v.tangentW.xyz = normalize(pIn.tangentW.xyz);
    v.tangentW.w = sign(pIn.tangentW.w); // Preserve zero to indicate invalid tangent.
    return shadeGBuffer(v, pIn.materialID, pIn.posH, pIn.prevPosH, instanceID, pIn.triangleIndex, pIn.bary, ddx(pIn.texC), ddy(pIn.texC));
}
#else
GBufferPSOut psMain(VSOut vsOut)
{
    float3 faceNormal = normalize(cross(ddy(vsOut.posW), ddx(vsOut.posW)));
    VertexData v = prepareVertexData(vsOut, faceNormal);
    return shadeGBuffer(v, vsOut.materialID, vsOut.posH, vsOut.prevPosH, vsOut.instanceID, 0, float3(0.f), ddx(vsOut.texC), ddy(vsOut.texC));
}
#endif
