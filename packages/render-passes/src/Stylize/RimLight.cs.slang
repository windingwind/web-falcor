/**
 * RimLight (see RimLight.ts). out = src + intensity * rim * colour * lerp(1, albedo, tint) on geometry pixels.
 *  mode 0 (fresnel)      : rim = (1 - N.V)^power * lerp(1, saturate(N.up), upBias)
 *  mode 1 (depth_offset) : rim = 1 where the depth sampled widthPx pixels along the screen-space normal is farther by more
 *                          than depthThreshold (relative) or is background; a constant-width screen-space rim.
 * Both are scaled by lerp(1, saturate(N.L), lightFacing).
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint gMode;
    float gIntensity;
    float3 gColour;
    float gPower;
    float3 gUp;
    float gUpBias;
    float3 gCamPos;
    float gWidthPx;
    float3 gCamFwd;
    float gDepthThreshold;
    float3 gLightDir;
    float gLightFacing;
    float4 gViewRow0;
    float4 gViewRow1;
    float gTint;
    uint gHasAlbedo;
};

Texture2D<float4> gSrc;
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gAlbedo;
RWTexture2D<float4> gDst;

float viewDepth(float3 posW) { return dot(posW - gCamPos, gCamFwd); }

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const int2 p = int2(tid.xy);
    if (any(p >= int2(gFrameDim))) return;
    const float4 src = gSrc[p];
    const float4 pos = gPosW[p];
    if (pos.w == 0.f || gIntensity == 0.f)
    {
        gDst[p] = src;
        return;
    }
    const float3 n = normalize(gNormW[p].xyz);
    float rim = 0.f;
    if (gMode == 0)
    {
        const float3 V = normalize(gCamPos - pos.xyz);
        rim = pow(1.f - saturate(dot(n, V)), gPower) * lerp(1.f, saturate(dot(n, gUp)), gUpBias);
    }
    else
    {
        // Screen-space direction of the normal (view-space x right, y up; screen y points down).
        float2 dir = float2(dot(gViewRow0.xyz, n), -dot(gViewRow1.xyz, n));
        const float len = length(dir);
        if (len > 1e-4f)
        {
            dir /= len;
            const int2 q = p + int2(round(dir * gWidthPx));
            if (any(q < 0) || any(q >= int2(gFrameDim)))
                rim = 1.f;
            else
            {
                const float4 posQ = gPosW[q];
                const float dp = viewDepth(pos.xyz);
                rim = (posQ.w == 0.f || viewDepth(posQ.xyz) - dp > gDepthThreshold * dp) ? 1.f : 0.f;
            }
        }
    }
    rim *= lerp(1.f, saturate(dot(n, gLightDir)), gLightFacing);
    float3 c = gColour;
    if (gHasAlbedo != 0) c *= lerp(float3(1.f), gAlbedo[p].rgb, gTint);
    gDst[p] = float4(src.rgb + gIntensity * rim * c, src.a);
}
