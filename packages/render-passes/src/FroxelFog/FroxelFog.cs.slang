/**
 * FroxelFog dispatches 2 and 3 of 3.
 *  integrate: one thread per froxel column, front-to-back over the slices with the energy-conserving per-slice
 *             integral (S - S T) / sigma_t (Hillaire 2015); writes gFroxelDst = (accumulated in-scatter, transmittance)
 *             at each slice's far edge.
 *  apply    : per pixel, the integrated volume at the pixel's view depth (trilinear; background pixels at the far
 *             distance when sky fog is on): dst = color * T + S; optional fog output (S, T).
 * Adapted in part from Godot Engine volumetric_fog_process.glsl (MIT, see FroxelFogCommon.slangh for the notice).
 */
#include "RenderPasses/FroxelFog/FroxelFogCommon.slangh"

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gFog;

[numthreads(8, 8, 1)]
void integrate(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gGrid.xy)) return;
    const float2 uv = (float2(tid.xy) + 0.5f) / float2(gGrid.xy);
    const float pathScale = length(cameraRay(uv)) / gCamPos.w; // path length per unit of view depth
    float3 acc = float3(0.f);
    float T = 1.f;
    float prevEdge = 0.f;
    for (uint z = 0; z < gGrid.z; z++)
    {
        const float edge = sliceDepth(float(z + 1) / float(gGrid.z));
        const float t = (edge - prevEdge) * pathScale;
        prevEdge = edge;
        const uint i = froxelIndex(uint3(tid.xy, z));
        const float4 m = unpackFroxel(gFroxelSrc[i]);
        const float Ts = exp(-m.a * t);
        const float3 S = m.a > 1e-7f ? (m.rgb - m.rgb * Ts) / m.a : m.rgb * t;
        acc += T * S;
        T *= Ts;
        gFroxelDst[i] = packFroxel(float4(acc, T));
    }
}

/// Integrated (S, T) at screen uv and view depth d.
float4 lookupFog(float2 uv, float d)
{
    const float s = d <= 0.f ? 0.f : float(gGrid.z) * pow(min(d / gSlices.x, 1.f), gSlices.z); // continuous slice-edge coordinate
    const float w = s - 1.f; // element z holds slice edge z + 1
    const float2 xy = uv * float2(gGrid.xy) - 0.5f;
    if (w >= 0.f) return sampleFroxels(float3(xy, w));
    const float4 v0 = sampleFroxels(float3(xy, 0.f));
    const float4 e = float4(0.f, 0.f, 0.f, 1.f);
    return e + (v0 - e) * (w + 1.f);
}

[numthreads(16, 16, 1)]
void apply(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pixel = tid.xy;
    if (any(pixel >= gMisc.xy)) return;
    const float4 c = gColor[pixel];
    const float4 pw = gPosW[pixel];
    float4 fog = float4(0.f, 0.f, 0.f, 1.f);
    const float2 uv = (float2(pixel) + 0.5f) / float2(gMisc.xy);
    if (pw.w != 0.f) fog = lookupFog(uv, dot(pw.xyz - gCamPos.xyz, gCamW.xyz) / gCamPos.w);
    else if ((gGrid.w & 2u) != 0u) fog = lookupFog(uv, gSlices.x);
    gDst[pixel] = float4(c.rgb * fog.a + fog.rgb, c.a);
    gFog[pixel] = fog;
}
