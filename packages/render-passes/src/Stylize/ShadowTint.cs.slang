/**
 * ShadowTint (see ShadowTint.ts). s = saturate(N.L) * visibility; m = 1 - smoothstep(threshold - softness, threshold + softness, s);
 * shadow colour = albedo in OKLCh with hue rotated by hueShift, lightness * (1 - valueDrop), chroma * satKeep, times
 * lerp(nearTint, farTint, smoothstep(near, far, view depth)) and `brightness`; out = lerp(src, shadow colour, m * strength).
 */
cbuffer CB
{
    uint2 gFrameDim;
    float gThreshold;
    float gSoftness;
    float3 gLightDir;
    float gHueShift;     ///< radians
    float3 gNearTint;
    float gValueDrop;
    float3 gFarTint;
    float gSatKeep;
    float3 gCamPos;
    float gNear;
    float3 gCamFwd;
    float gFar;
    float gBrightness;
    float gStrength;
    uint gHasVisibility;
};
Texture2D<float4> gSrc;
Texture2D<float4> gAlbedo;
Texture2D<float4> gNormW;
Texture2D<float4> gPosW;
Texture2D<float4> gVisibility;
RWTexture2D<float4> gDst;

float cbrtS(float x) { return sign(x) * pow(abs(x), 1.f / 3.f); }
float3 toOklab(float3 c)
{
    const float l = cbrtS(0.4122214708f * c.r + 0.5363325363f * c.g + 0.0514459929f * c.b);
    const float m = cbrtS(0.2119034982f * c.r + 0.6806995451f * c.g + 0.1073969566f * c.b);
    const float s = cbrtS(0.0883024619f * c.r + 0.2817188376f * c.g + 0.6299787005f * c.b);
    return float3(0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s, 1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
                  0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s);
}
float3 fromOklab(float3 L)
{
    const float l = L.x + 0.3963377774f * L.y + 0.2158037573f * L.z;
    const float m = L.x - 0.1055613458f * L.y - 0.0638541728f * L.z;
    const float s = L.x - 0.0894841775f * L.y - 1.2914855480f * L.z;
    const float l3 = l * l * l, m3 = m * m * m, s3 = s * s * s;
    return float3(4.0767416621f * l3 - 3.3077115913f * m3 + 0.2309699292f * s3, -1.2684380046f * l3 + 2.6097574011f * m3 - 0.3413193965f * s3,
                  -0.0041960863f * l3 - 0.7034186147f * m3 + 1.7076147010f * s3);
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 src = gSrc[p];
    const float4 pos = gPosW[p];
    if (pos.w == 0.f || gStrength == 0.f) { gDst[p] = src; return; }
    float s = saturate(dot(normalize(gNormW[p].xyz), gLightDir));
    if (gHasVisibility != 0) s *= saturate(gVisibility[p].r);
    const float m = 1.f - smoothstep(gThreshold - gSoftness, gThreshold + gSoftness + 1e-5f, s);
    float3 lab = toOklab(max(gAlbedo[p].rgb, 0.f));
    const float cs = cos(gHueShift), sn = sin(gHueShift);
    lab.yz = float2(cs * lab.y - sn * lab.z, sn * lab.y + cs * lab.z) * gSatKeep;
    lab.x *= 1.f - gValueDrop;
    const float d = dot(pos.xyz - gCamPos, gCamFwd);
    const float3 tint = lerp(gNearTint, gFarTint, smoothstep(gNear, max(gFar, gNear + 1e-4f), d));
    const float3 shadow = max(fromOklab(lab), 0.f) * tint * gBrightness;
    gDst[p] = float4(lerp(src.rgb, shadow, m * gStrength), src.a);
}
