/**
 * WebFalcor override of RenderPasses/GBuffer/GBuffer/GBufferRaster.3d.slang
 * (docs §4.3/§9). Diffs vs upstream: psMain drops SV_PrimitiveID and
 * SV_Barycentrics (absent from WGSL) and [earlydepthstencil]; the face normal
 * comes from screen-space derivatives. WebGPU has no rasterizer-ordered
 * views, so the "extra" UAV channels are render targets too: the host assigns
 * every connected channel a batch + location (LOC_<texname>, is_valid_<texname>)
 * and draws one geometry pass per batch of <= 8 targets. The vbuffer channel
 * needs the primitive id and is served by VBufferRaster instead.
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
};

VSOut vsMain(VSIn vsIn)
{
    return defaultVS(vsIn);
}

float2 computeMotionVector(const VSOut vsOut, const int2 ipos)
{
    // Current sample in pixel coords.
    float2 pixelPos = ipos + float2(0.5, 0.5);
    // Sample in previous frame in clip space coords, no jittering applied.
    float4 prevPosH = vsOut.prevPosH;
    // Remove camera jitter from motion vector
    return calcMotionVector(pixelPos, prevPosH, gFrameDim) + float2(gScene.camera.data.jitterX, -gScene.camera.data.jitterY);
}

// WebFalcor: SV_PrimitiveID/SV_Barycentrics are unavailable on the WGSL target
// ([earlydepthstencil] likewise). The face normal is derived from position
// derivatives instead of indexed geometry -- equivalent for planar triangles.
GBufferPSOut psMain(VSOut vsOut)
{
    // Using vOut.posH.xy as pixel coordinate since it has the SV_Position semantic.
    int2 ipos = int2(vsOut.posH.xy);

    float3 faceNormal = normalize(cross(ddy(vsOut.posW), ddx(vsOut.posW)));
    VertexData v = prepareVertexData(vsOut, faceNormal);
    // WebFalcor: WGSL requires uniform control flow for implicit-LOD sampling;
    // hoist uv derivatives here and sample by explicit gradients (equivalent).
    let lod = ExplicitGradientTextureSampler(ddx(vsOut.texC), ddy(vsOut.texC));

#if USE_ALPHA_TEST
    if (gScene.materials.alphaTest(v, vsOut.materialID, lod))
        discard;
#endif
    const float3 viewDir = normalize(gScene.camera.getPosition() - v.posW);
    ShadingData sd = gScene.materials.prepareShadingData(v, vsOut.materialID, viewDir);

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
    psOut.mvec = computeMotionVector(vsOut, ipos);
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
        const float linearZ = vsOut.posH.z * vsOut.posH.w;
        psOut.linearZ = float2(linearZ, max(abs(ddx(linearZ)), abs(ddy(linearZ))));
    }
#endif

    return psOut;
}
