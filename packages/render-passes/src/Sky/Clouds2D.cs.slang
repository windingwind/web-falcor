/**
 * Clouds2D kernel (see Clouds2D.ts; its evalClouds2D() is the CPU reference of this shader).
 * Value-noise fBm cloud plane with one domain warp, 2D self-shadow towards the light, silver lining. Up axis +y.
 */

cbuffer CB
{
    uint2 gFrameDim;
    uint gHasColor;
    uint gHasPos;
    uint gSalt;
    uint gOctaves;
    uint gHasLayer;
    uint gHasLayerDepth;
    float4 gCamRight;   ///< xyz = right * tan(fovY/2) * aspect, w = coverage
    float4 gCamUp;      ///< xyz = up * tan(fovY/2), w = softness
    float4 gCamFwd;     ///< xyz = forward, w = horizonFade
    float4 gPlane;      ///< height, curvature, scale, warp
    float4 gWind;       ///< wind * time (xy), opacity, brightness
    float4 gLight;      ///< xyz = towards the light, w = silver
    float4 gLight2;     ///< light direction on the plane (xz, normalised), shadowStep, absorption
    float4 gLit;
    float4 gShade;
};

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;
RWTexture2D<float4> gLayer;
RWTexture2D<float4> gLayerDepth;

static const float kCloudDepth = 2.9e38f;

uint3 pcg3d(uint3 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> 16u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

float u01(uint u) { return float(u >> 8u) * (1.f / 16777216.f); }

float vnoise2(float2 p, uint salt)
{
    const float2 fi = floor(p);
    const int2 i = int2(fi);
    const float2 f = p - fi;
    const float2 w = f * f * (3.f - 2.f * f);
    const float v00 = u01(pcg3d(uint3(uint(i.x), uint(i.y), salt)).x);
    const float v10 = u01(pcg3d(uint3(uint(i.x + 1), uint(i.y), salt)).x);
    const float v01 = u01(pcg3d(uint3(uint(i.x), uint(i.y + 1), salt)).x);
    const float v11 = u01(pcg3d(uint3(uint(i.x + 1), uint(i.y + 1), salt)).x);
    return lerp(lerp(v00, v10, w.x), lerp(v01, v11, w.x), w.y);
}

float fbm2(float2 p, uint oct, uint salt)
{
    float s = 0.f, a = 0.5f, n = 0.f;
    for (uint o = 0; o < oct; o++)
    {
        s += a * vnoise2(p, salt + o);
        n += a;
        p = 2.02f * float2(0.8f * p.x - 0.6f * p.y, 0.6f * p.x + 0.8f * p.y);
        a *= 0.5f;
    }
    return s / n;
}

float density(float2 uv)
{
    const float w = fbm2(0.5f * uv + float2(3.1f, 7.7f), 3, gSalt + 20u) - 0.5f;
    const float n = fbm2(uv + gPlane.w * float2(w, -w), gOctaves, gSalt + 40u);
    return saturate((n - (1.f - gCamRight.w)) / gCamUp.w);
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 src = gHasColor != 0 ? gColor[pix] : float4(0.f, 0.f, 0.f, 1.f);
    float4 o = src;
    float4 lay = float4(0.f);
    const bool geometry = gHasPos != 0 && gPosW[pix].w != 0.f;
    if (!geometry && gCamRight.w > 0.f && gWind.z > 0.f)
    {
        const float2 ndc = (float2(pix) + 0.5f) / float2(gFrameDim) * 2.f - 1.f;
        const float3 d = normalize(gCamFwd.xyz + ndc.x * gCamRight.xyz - ndc.y * gCamUp.xyz);
        if (d.y > 0.f)
        {
            const float k = gPlane.x / (d.y + gPlane.y);
            const float2 uv = d.xz * k * gPlane.z + gWind.xy;
            const float dens = density(uv);
            const float a = dens * gWind.z * smoothstep(0.f, gCamFwd.w, d.y);
            if (a > 0.f)
            {
                const float ds = density(uv + gLight2.xy * gLight2.z);
                const float T = exp(-gLight2.w * ds);
                const float sil = gLight.w * pow(max(dot(d, gLight.xyz), 0.f), 8.f) * (1.f - dens);
                const float3 rgb = (lerp(gShade.rgb, gLit.rgb, T) + sil * gLit.rgb) * gWind.w * a;
                lay = float4(rgb, a);
                o.rgb = src.rgb * (1.f - a) + rgb;
            }
        }
    }
    gDst[pix] = o;
    if (gHasLayer != 0) gLayer[pix] = lay;
    if (gHasLayerDepth != 0) gLayerDepth[pix] = float4(kCloudDepth, 0.f, 0.f, 1.f);
}
