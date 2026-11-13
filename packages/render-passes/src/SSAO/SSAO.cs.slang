/** SSAO (GTAO estimator) kernels for SSAO.ts.
 *  aoMain  : per pixel, gSliceCount slices through the view vector; per slice the two horizon angles are found by
 *            marching gSteps samples each way in screen space (world radius gRadius projected, capped at
 *            gMaxPixelRadius); the cosine-weighted visible arc is integrated analytically against the normal
 *            projected into the slice (Jimenez et al. 2016, eq. 10; same arrangement as XeGTAO).
 *  blurMain: 5x5 spatial filter weighted by view-depth and normal similarity.
 *  Conventions: Falcor view space (right-handed, camera looks down -z), pixel y grows downwards.
 *  Only Load() is used (no samplers), so 32-bit float inputs bind as unfilterable-float. */

static const float kPi = 3.14159265358979f;
static const float kHalfPi = 1.57079632679490f;

cbuffer CB
{
    uint2 gFrameDim;
    uint gSliceCount;
    uint gSteps;
    float4 gViewRow0;
    float4 gViewRow1;
    float4 gViewRow2;
    float gFocalPx;        // focal length in pixels (H / (2 tan(fovY/2)))
    float gRadius;         // world-space occluder radius
    float gFalloff;        // fraction of the radius over which occluder weight fades to 0
    float gMaxPixelRadius;
    float gPower;
    uint gNoise;
    uint gFrame;
    float _pad0;
}

Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gRaw;
RWTexture2D<float4> gOut;

float3 toView(float3 p) { float4 q = float4(p, 1.f); return float3(dot(gViewRow0, q), dot(gViewRow1, q), dot(gViewRow2, q)); }
float3 dirToView(float3 d) { float4 q = float4(d, 0.f); return float3(dot(gViewRow0, q), dot(gViewRow1, q), dot(gViewRow2, q)); }

/** Interleaved gradient noise (Jimenez 2014). */
float ign(float2 p) { return frac(52.9829189f * frac(dot(p, float2(0.06711056f, 0.00583715f)))); }

float fastAcos(float x)
{
    x = clamp(x, -1.f, 1.f);
    return acos(x);
}

[numthreads(16, 16, 1)]
void aoMain(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 pw = gPosW[pix];
    if (pw.w == 0.f) { gOut[pix] = float4(1.f); return; }

    const float3 P = toView(pw.xyz);
    const float3 V = normalize(-P);
    float3 N = normalize(dirToView(gNormW[pix].xyz));
    if (dot(N, V) < 0.f) N = -N;

    const float depth = max(-P.z, 1e-4f);
    const float rPx = min(gRadius * gFocalPx / depth, gMaxPixelRadius);
    if (rPx < 1.f) { gOut[pix] = float4(1.f); return; }
    const float falloffRange = max(gFalloff * gRadius, 1e-4f);
    const float falloffFrom = gRadius - falloffRange;

    const float2 center = float2(pix) + 0.5f;
    const float noiseSlice = gNoise != 0 ? ign(center) : 0.5f;
    const float noiseStep = gNoise != 0 ? ign(center + float2(5.588238f, 2.0f)) : 0.5f;

    float vis = 0.f;
    for (uint s = 0; s < gSliceCount; s++)
    {
        const float phi = (float(s) + noiseSlice) / float(gSliceCount) * kPi;
        const float2 omega = float2(cos(phi), sin(phi));        // screen direction, y down
        const float3 dirV = float3(omega.x, -omega.y, 0.f);       // the same direction in view space (y up)
        const float3 orthoDir = dirV - dot(dirV, V) * V;
        const float3 axis = normalize(cross(orthoDir, V));
        const float3 projN = N - axis * dot(N, axis);
        const float projNLen = length(projN);
        if (projNLen < 1e-5f) continue;
        const float sgnN = dot(orthoDir, projN) >= 0.f ? 1.f : -1.f;
        const float cosN = clamp(dot(projN, V) / projNLen, -1.f, 1.f);
        const float n = sgnN * fastAcos(cosN);

        const float lowCos0 = cos(n + kHalfPi); // +omega side
        const float lowCos1 = cos(n - kHalfPi); // -omega side
        float hc0 = lowCos0, hc1 = lowCos1;
        for (uint k = 0; k < gSteps; k++)
        {
            float t = (float(k) + noiseStep) / float(gSteps);
            t = t * t;
            const float off = 1.f + t * (rPx - 1.f);
            [unroll]
            for (uint side = 0; side < 2; side++)
            {
                const float2 q = center + (side == 0 ? omega : -omega) * off;
                if (any(q < 0.f) || any(q >= float2(gFrameDim))) continue;
                const float4 sw = gPosW[uint2(q)];
                if (sw.w == 0.f) continue;
                const float3 d = toView(sw.xyz) - P;
                const float dist = length(d);
                if (dist < 1e-6f) continue;
                const float shc = dot(d / dist, V);
                const float w = saturate((gRadius - dist) / falloffRange);
                if (side == 0) hc0 = max(hc0, lerp(lowCos0, shc, w));
                else hc1 = max(hc1, lerp(lowCos1, shc, w));
            }
        }
        // horizon angles measured from V towards +omega (h1) and -omega (h0), clamped to the normal's hemisphere
        float h1 = fastAcos(hc0);
        float h0 = -fastAcos(hc1);
        h0 = n + clamp(h0 - n, -kHalfPi, kHalfPi);
        h1 = n + clamp(h1 - n, -kHalfPi, kHalfPi);
        const float sinN = sin(n);
        const float arc0 = (cosN + 2.f * h0 * sinN - cos(2.f * h0 - n)) * 0.25f;
        const float arc1 = (cosN + 2.f * h1 * sinN - cos(2.f * h1 - n)) * 0.25f;
        vis += projNLen * (arc0 + arc1);
    }
    vis = saturate(vis / float(gSliceCount));
    vis = pow(vis, gPower);
    gOut[pix] = float4(vis, vis, vis, 1.f);
}

[numthreads(16, 16, 1)]
void blurMain(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    const float4 pw = gPosW[pix];
    if (pw.w == 0.f) { gOut[pix] = float4(1.f); return; }
    const float z0 = -toView(pw.xyz).z;
    const float3 n0 = gNormW[pix].xyz;
    float sum = 0.f, wsum = 0.f;
    for (int dy = -2; dy <= 2; dy++)
    {
        for (int dx = -2; dx <= 2; dx++)
        {
            const int2 q = int2(pix) + int2(dx, dy);
            if (any(q < 0) || any(q >= int2(gFrameDim))) continue;
            const float4 sw = gPosW[uint2(q)];
            if (sw.w == 0.f) continue;
            const float z = -toView(sw.xyz).z;
            const float wz = exp(-abs(z - z0) / max(0.02f * z0, 1e-4f));
            const float wn = pow(saturate(dot(gNormW[uint2(q)].xyz, n0)), 8.f);
            const float w = wz * wn;
            sum += w * gRaw[uint2(q)].x;
            wsum += w;
        }
    }
    const float v = wsum > 0.f ? sum / wsum : gRaw[pix].x;
    gOut[pix] = float4(v, v, v, 1.f);
}
