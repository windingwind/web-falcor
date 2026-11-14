/**
 * EdgeAwareFilter (see EdgeAwareFilter.ts): edge-preserving smoothing / detail manipulation, three methods.
 *
 * Written from the papers (no code taken from other implementations):
 *  - guided filter: K. He, J. Sun, X. Tang, "Guided Image Filtering", ECCV 2010 (grey guide, per-channel input);
 *  - bilateral grid: J. Chen, S. Paris, F. Durand, "Real-time Edge-Aware Image Processing with the Bilateral Grid",
 *    SIGGRAPH 2007 (nearest-cell splat, separable grid blur, trilinear slice);
 *  - fast local Laplacian filter: S. Paris, S. W. Hasinoff, J. Kautz, "Local Laplacian Filters", SIGGRAPH 2011, and
 *    M. Aubry, S. Paris, S. W. Hasinoff, J. Kautz, F. Durand, "Fast Local Laplacian Filters", ACM TOG 2014
 *    (sampled reference intensities, interpolation of the remapped Laplacian coefficients).
 *
 * Guide value g: gHasGuide ? gGuideScale * guide.x : (luminance)^(1/2.2) (perceptually more even range distances).
 * Guided filter / bilateral grid: base = filtered colour; result = src + detail * (src - base).
 * Local Laplacian: remap r(i) = i + detail * d * exp(-d^2 / (2 sigmaR^2)), d = i - gamma, on the encoded luminance;
 * the colour is scaled by the luminance ratio. Output = src + strength * (result - src), clamped to >= 0.
 */

cbuffer CB
{
    uint2 gFrameDim;      ///< full-resolution size
    uint2 gInDim;         ///< local Laplacian: size of the level read (fine level for llDown, coarse level for llLap/llCollapse/llFinal)
    uint2 gOutDim;        ///< local Laplacian: size of the level written
    int gRadius;          ///< guided filter window radius (px)
    float gEps;           ///< guided filter regulariser (guide units squared)
    float gDetail;
    float gStrength;
    uint gHasGuide;
    float gGuideScale;
    uint gGridW;          ///< bilateral grid size (cells)
    uint gGridH;
    uint gGridD;
    uint gCellSize;       ///< bilateral grid spatial cell size (px)
    float gInvSigmaR;     ///< 1 / sigmaR (bilateral grid range cell size, local Laplacian detail range)
    uint gAxis;           ///< bilateral grid blur axis 0/1/2
    float gRef;           ///< local Laplacian reference intensity gamma_k
    float gRefStep;       ///< (number of reference intensities - 1)
    uint gFirst;          ///< local Laplacian: first reference intensity (overwrite instead of accumulate)
};

Texture2D<float4> gSrc;
Texture2D<float4> gGuide;
RWTexture2D<float4> gDst;

// guided filter scratch
Texture2D<float4> gInA;
Texture2D<float4> gInB;
RWTexture2D<float4> gOutA;
RWTexture2D<float4> gOutB;

// bilateral grid: 4 x u32 per cell (r, g, b, weight) in fixed point, then float4 cells
RWStructuredBuffer<Atomic<uint>> gGridU;
StructuredBuffer<float4> gGridIn;
RWStructuredBuffer<float4> gGridOut;

// local Laplacian pyramid levels
Texture2D<float> gInL;
Texture2D<float> gCoarseL;
Texture2D<float> gGuideL;
RWTexture2D<float> gOutL;
RWStructuredBuffer<float> gAcc;

static const float3 kLumaW = float3(0.2126f, 0.7152f, 0.0722f);
static const float kFix = 1024.f;     ///< bilateral grid fixed-point scale (1/1024)
static const float kSplatMax = 64.f;  ///< colour values are clamped to [0, 64] for the splat (keeps 64x64-px cells within 32 bits)
static const float kBinom[5] = { 1.f / 16.f, 4.f / 16.f, 6.f / 16.f, 4.f / 16.f, 1.f / 16.f };

float encodedLuma(float3 c)
{
    const float l = dot(c, kLumaW);
    return l > 0.f ? pow(l, 1.f / 2.2f) : 0.f;
}

float guideAt(int2 p)
{
    return gHasGuide != 0 ? gGuideScale * gGuide[p].x : encodedLuma(gSrc[p].rgb);
}

float4 compose(float4 s, float3 result)
{
    return float4(max(s.rgb + gStrength * (result - s.rgb), 0.f), s.a);
}

bool outside(uint2 p, uint2 dim)
{
    return any(p >= dim);
}

// ---------------------------------------------------------------------------------------------------------------------
// identity (detail = 0 or strength = 0)

[numthreads(16, 16, 1)]
void copyMain(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    gDst[tid.xy] = gSrc[tid.xy];
}

// ---------------------------------------------------------------------------------------------------------------------
// guided filter: box sums over the window truncated at the image border, divided by the pixel count of the window.
//   gfH1: horizontal sums of (I, I^2, p.rgb, I p.rgb)  -> A = (I, I^2, p.r, p.g), B = (p.b, I p.r, I p.g, I p.b)
//   gfV1: vertical sums -> means -> a = cov(I, p) / (var(I) + eps), b = mean(p) - a mean(I)   -> A = (a, 0), B = (b, 0)
//   gfH2: horizontal sums of a, b
//   gfV2: vertical sums -> mean a, mean b -> base = mean(a) I + mean(b) -> output

int2 windowRange(int c, int n)
{
    return int2(max(c - gRadius, 0), min(c + gRadius, n - 1));
}

[numthreads(16, 16, 1)]
void gfH1(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const int2 r = windowRange(p.x, int(gFrameDim.x));
    float4 a = 0.f, b = 0.f;
    for (int x = r.x; x <= r.y; x++)
    {
        const int2 q = int2(x, p.y);
        const float I = guideAt(q);
        const float3 c = gSrc[q].rgb;
        a += float4(I, I * I, c.r, c.g);
        b += float4(c.b, I * c.r, I * c.g, I * c.b);
    }
    gOutA[p] = a;
    gOutB[p] = b;
}

[numthreads(16, 16, 1)]
void gfV1(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const int2 rx = windowRange(p.x, int(gFrameDim.x));
    const int2 ry = windowRange(p.y, int(gFrameDim.y));
    float4 a = 0.f, b = 0.f;
    for (int y = ry.x; y <= ry.y; y++)
    {
        a += gInA[int2(p.x, y)];
        b += gInB[int2(p.x, y)];
    }
    const float inv = 1.f / float((rx.y - rx.x + 1) * (ry.y - ry.x + 1));
    a *= inv;
    b *= inv;
    const float meanI = a.x;
    const float varI = max(a.y - meanI * meanI, 0.f);
    const float3 meanP = float3(a.z, a.w, b.x);
    const float3 cov = b.yzw - meanI * meanP;
    const float3 ca = cov / (varI + gEps);
    gOutA[p] = float4(ca, 0.f);
    gOutB[p] = float4(meanP - ca * meanI, 0.f);
}

[numthreads(16, 16, 1)]
void gfH2(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const int2 r = windowRange(p.x, int(gFrameDim.x));
    float4 a = 0.f, b = 0.f;
    for (int x = r.x; x <= r.y; x++)
    {
        a += gInA[int2(x, p.y)];
        b += gInB[int2(x, p.y)];
    }
    gOutA[p] = a;
    gOutB[p] = b;
}

[numthreads(16, 16, 1)]
void gfV2(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const int2 rx = windowRange(p.x, int(gFrameDim.x));
    const int2 ry = windowRange(p.y, int(gFrameDim.y));
    float4 a = 0.f, b = 0.f;
    for (int y = ry.x; y <= ry.y; y++)
    {
        a += gInA[int2(p.x, y)];
        b += gInB[int2(p.x, y)];
    }
    const float inv = 1.f / float((rx.y - rx.x + 1) * (ry.y - ry.x + 1));
    const float4 s = gSrc[p];
    const float3 base = a.xyz * inv * guideAt(p) + b.xyz * inv;
    gDst[p] = compose(s, s.rgb + gDetail * (s.rgb - base));
}

// ---------------------------------------------------------------------------------------------------------------------
// bilateral grid. Cell of pixel p: (p.x / cellSize, p.y / cellSize, round(g / sigmaR)), g clamped to [0, 1].
//   bgClear: zero the u32 grid;  bgSplat: fixed-point atomic adds of (rgb, 1);
//   bgBlurU / bgBlur: [1 4 6 4 1] / 16 along x, then y, then z (cells outside the grid count as empty);
//   bgSlice: trilinear interpolation at the pixel's continuous grid position, base = rgb / weight.

uint cellIndex(uint x, uint y, uint z)
{
    return (z * gGridH + y) * gGridW + x;
}

float clampedGuide(int2 p)
{
    return clamp(guideAt(p), 0.f, 1.f);
}

[numthreads(64, 1, 1)]
void bgClear(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x < gGridW * gGridH * gGridD * 4) gGridU[tid.x].store(0u);
}

[numthreads(16, 16, 1)]
void bgSplat(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const uint z = min(uint(clampedGuide(p) * gInvSigmaR + 0.5f), gGridD - 1);
    const uint i = cellIndex(tid.x / gCellSize, tid.y / gCellSize, z) * 4;
    const float3 c = clamp(gSrc[p].rgb, 0.f, kSplatMax);
    gGridU[i + 0].add(uint(c.r * kFix + 0.5f));
    gGridU[i + 1].add(uint(c.g * kFix + 0.5f));
    gGridU[i + 2].add(uint(c.b * kFix + 0.5f));
    gGridU[i + 3].add(uint(kFix));
}

int3 cellCoord(uint idx)
{
    return int3(int(idx % gGridW), int((idx / gGridW) % gGridH), int(idx / (gGridW * gGridH)));
}

bool inGrid(int3 c)
{
    return all(c >= 0) && c.x < int(gGridW) && c.y < int(gGridH) && c.z < int(gGridD);
}

int3 axisStep()
{
    return gAxis == 0 ? int3(1, 0, 0) : gAxis == 1 ? int3(0, 1, 0) : int3(0, 0, 1);
}

[numthreads(64, 1, 1)]
void bgBlurU(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gGridW * gGridH * gGridD) return;
    const int3 c = cellCoord(tid.x);
    const int3 d = axisStep();
    float4 acc = 0.f;
    for (int t = -2; t <= 2; t++)
    {
        const int3 q = c + t * d;
        if (!inGrid(q)) continue;
        const uint j = cellIndex(q.x, q.y, q.z) * 4;
        const float4 v = float4(float(gGridU[j].load()), float(gGridU[j + 1].load()), float(gGridU[j + 2].load()), float(gGridU[j + 3].load()));
        acc += kBinom[t + 2] * (v / kFix);
    }
    gGridOut[tid.x] = acc;
}

[numthreads(64, 1, 1)]
void bgBlur(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gGridW * gGridH * gGridD) return;
    const int3 c = cellCoord(tid.x);
    const int3 d = axisStep();
    float4 acc = 0.f;
    for (int t = -2; t <= 2; t++)
    {
        const int3 q = c + t * d;
        if (inGrid(q)) acc += kBinom[t + 2] * gGridIn[cellIndex(q.x, q.y, q.z)];
    }
    gGridOut[tid.x] = acc;
}

[numthreads(16, 16, 1)]
void bgSlice(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const float4 s = gSrc[p];
    const float3 u = clamp(float3((float2(p) + 0.5f) / float(gCellSize) - 0.5f, clampedGuide(p) * gInvSigmaR),
                           float3(0.f), float3(float(gGridW - 1), float(gGridH - 1), float(gGridD - 1)));
    const uint3 c0 = uint3(floor(u));
    const uint3 c1 = min(c0 + 1, uint3(gGridW - 1, gGridH - 1, gGridD - 1));
    const float3 f = u - float3(c0);
    float4 acc = 0.f;
    for (uint k = 0; k < 8; k++)
    {
        const bool bx = (k & 1) != 0, by = (k & 2) != 0, bz = (k & 4) != 0;
        const float w = (bx ? f.x : 1.f - f.x) * (by ? f.y : 1.f - f.y) * (bz ? f.z : 1.f - f.z);
        acc += w * gGridIn[cellIndex(bx ? c1.x : c0.x, by ? c1.y : c0.y, bz ? c1.z : c0.z)];
    }
    const float3 base = acc.w > 1e-6f ? acc.rgb / acc.w : s.rgb;
    gDst[p] = compose(s, s.rgb + gDetail * (s.rgb - base));
}

// ---------------------------------------------------------------------------------------------------------------------
// fast local Laplacian on the encoded luminance u.
//   llGuide   : G0 = u
//   llDown    : next Gaussian level, [1 4 6 4 1] / 16 separable kernel at 2x, clamp to edge; size ceil(n / 2)
//   llRemap   : R0 = remap(G0, gamma_k)
//   llLap     : acc_l += hat_k(Gin_l) * (Rk_l - up(Rk_l+1))   (hat weight over the reference intensities)
//   llCollapse: out_l = acc_l + up(out_l+1)   (out_n = Gin_n)
//   llFinal   : level 0 collapse + colour reconstruction by the luminance ratio
// up(): bilinear interpolation of the coarse level at ((p + 0.5) / 2 - 0.5), clamped to its extent.

float upsample(Texture2D<float> coarse, int2 p)
{
    const float2 c = clamp(float2(p) * 0.5f - 0.25f, float2(0.f), float2(gInDim) - 1.f);
    const int2 c0 = int2(floor(c));
    const int2 c1 = min(c0 + 1, int2(gInDim) - 1);
    const float2 f = c - float2(c0);
    const float top = lerp(coarse[c0], coarse[int2(c1.x, c0.y)], f.x);
    const float bot = lerp(coarse[int2(c0.x, c1.y)], coarse[c1], f.x);
    return lerp(top, bot, f.y);
}

[numthreads(16, 16, 1)]
void llGuide(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    gOutL[tid.xy] = encodedLuma(gSrc[tid.xy].rgb);
}

[numthreads(16, 16, 1)]
void llDown(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gOutDim)) return;
    const int2 q = int2(tid.xy) * 2;
    float acc = 0.f;
    for (int j = -2; j <= 2; j++)
    {
        float row = 0.f;
        for (int i = -2; i <= 2; i++) row += kBinom[i + 2] * gInL[clamp(q + int2(i, j), int2(0), int2(gInDim) - 1)];
        acc += kBinom[j + 2] * row;
    }
    gOutL[tid.xy] = acc;
}

[numthreads(16, 16, 1)]
void llRemap(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const float g = gInL[tid.xy];
    const float d = g - gRef;
    const float t = d * gInvSigmaR;
    gOutL[tid.xy] = g + gDetail * d * exp(-0.5f * t * t);
}

[numthreads(16, 16, 1)]
void llLap(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gOutDim)) return;
    const int2 p = int2(tid.xy);
    const float lap = gGuideL[p] - upsample(gCoarseL, p);   // gGuideL = remapped level l here, gCoarseL = remapped level l + 1
    const float w = max(1.f - abs(clamp(gInL[p], 0.f, 1.f) - gRef) * gRefStep, 0.f); // gInL = input Gaussian level l
    const uint i = tid.y * gOutDim.x + tid.x;
    gAcc[i] = (gFirst != 0 ? 0.f : gAcc[i]) + w * lap;
}

[numthreads(16, 16, 1)]
void llCollapse(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gOutDim)) return;
    gOutL[tid.xy] = gAcc[tid.y * gOutDim.x + tid.x] + upsample(gCoarseL, int2(tid.xy));
}

[numthreads(16, 16, 1)]
void llFinal(uint3 tid: SV_DispatchThreadID)
{
    if (outside(tid.xy, gFrameDim)) return;
    const int2 p = int2(tid.xy);
    const float4 s = gSrc[p];
    const float u = max(gAcc[tid.y * gFrameDim.x + tid.x] + upsample(gCoarseL, p), 0.f);
    const float l = dot(s.rgb, kLumaW);
    const float l2 = u > 0.f ? pow(u, 2.2f) : 0.f;
    const float3 result = l > 1e-5f ? s.rgb * (l2 / l) : s.rgb + (l2 - l);
    gDst[p] = compose(s, result);
}
