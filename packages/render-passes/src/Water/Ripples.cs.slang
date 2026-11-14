/**
 * Ripples: a world-anchored water height field (2D wave equation) driven by rain drops, point emitters, moving objects
 * and scene objects in contact with the water plane; applied as a normal perturbation to the water surface pixels
 * (see Ripples.ts). Entry points:
 *  clearContact (per cell)  : zero the contact counters
 *  contact      (per pixel) : pixels of the chosen instances within contactDistance of the plane mark their cell
 *  step         (per cell)  : one wave-equation step on the (h, h_prev) ping-pong texture + all drivers
 *  apply        (per pixel) : bilinear field gradient -> perturbed normal on the water pixels
 *
 * Height update (step) adapted from the three.js example webgpu_compute_water (MIT licence,
 * https://github.com/mrdoob/three.js/blob/dev/examples/webgpu_compute_water.html, Copyright (c) 2010-2026 three.js
 * authors): h' = ((h_N + h_S + h_E + h_W) / 2 - h_prev) x damping + drivers, with the raised-cosine bump
 * 0.5 (1 + cos(pi d / r)) for impacts. Drivers, contact marking, sponge border and the screen-space application were
 * written for this pass.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
 * persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice
 * and this permission notice shall be included in all copies or substantial portions of the Software.
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
cbuffer CB
{
    uint2 gFrameDim;
    uint gGrid;            ///< N (cells per side)
    uint gStep;            ///< index of this step since the last reset
    float3 gUp;
    float gLevel;          ///< plane height along gUp
    float3 gT1;
    float gSize;           ///< region edge (world units)
    float3 gT2;
    float gDamping;
    float2 gCentre;        ///< region centre in planar coordinates (t1, t2)
    float gStepTime;       ///< seconds per step
    uint gSeed;
    uint gDrops;           ///< rain drops per step
    float gDropRadius;     ///< world units
    float gDropStrength;
    uint gEmitterCount;
    float4 gEmitterA[8];   ///< planar x, y, radius, strength
    float4 gEmitterB[8];   ///< period, phase
    uint gMoverCount;
    float gContactStrength;
    float gContactPeriod;
    float gContactDistance;
    float4 gMoverA[8];     ///< from x, y, to x, y (planar)
    float4 gMoverB[8];     ///< t0, t1, radius, strength
    uint4 gContactIds0;
    uint4 gContactIds1;
    uint gContactCount;
    uint gApplyContact;
    float gStrength;
    float gPlaneTolerance;
    float gUpMin;
    float gEdgeFade;
    uint gHasMask;
    uint gMaskChannel;
    float gSponge;         ///< border cells with extra damping
};

Texture2D<float4> gRead;          ///< (h, h_prev) of the previous step
RWTexture2D<float4> gWrite;
Texture2D<float4> gField;         ///< latest field (apply)
RWStructuredBuffer<Atomic<uint>> gContact;
Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gIds;
Texture2D<float4> gMask;
RWTexture2D<float4> gNormWOut;
RWTexture2D<float4> gHeightOut;

uint pcg(uint v)
{
    const uint state = v * 747796405u + 2891336453u;
    const uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rnd(uint a, uint b, uint salt)
{
    return float(pcg(a ^ pcg(b * 9781u + pcg(gSeed * 6271u + salt))) >> 8) * (1.f / 16777216.f);
}

/** Raised-cosine bump of radius r (cells) at distance d (cells). */
float bump(float d, float r)
{
    return d < r ? 0.5f * (1.f + cos(3.14159265f * d / r)) : 0.f;
}

float cellSize()
{
    return gSize / float(gGrid);
}

/** Planar world coordinates -> continuous cell coordinates (cell centres at i + 0.5). */
float2 toCell(float2 q)
{
    return (q - gCentre) / cellSize() + 0.5f * float(gGrid);
}

[numthreads(16, 16, 1)]
void clearContact(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= uint2(gGrid))) return;
    gContact[tid.y * gGrid + tid.x].store(0u);
}

bool chosen(uint id)
{
    for (uint k = 0; k < gContactCount; k++)
    {
        const uint c = k < 4u ? gContactIds0[k] : gContactIds1[k - 4u];
        if (c == id) return true;
    }
    return false;
}

[numthreads(16, 16, 1)]
void contact(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 x = gPosW[p];
    if (x.w == 0.f) return;
    if (!chosen(uint(gIds[p].r + 0.5f))) return;
    if (abs(dot(x.xyz, gUp) - gLevel) > gContactDistance) return;
    const float2 c = toCell(float2(dot(x.xyz, gT1), dot(x.xyz, gT2)));
    if (any(c < 0.f) || any(c >= float(gGrid))) return;
    const uint2 ci = uint2(c);
    gContact[ci.y * gGrid + ci.x].add(1u);
}

[numthreads(16, 16, 1)]
void step(uint3 tid: SV_DispatchThreadID)
{
    const int2 c = int2(tid.xy);
    const int N = int(gGrid);
    if (any(c >= int2(N))) return;
    const float2 s = gRead[c].xy;
    const float hN = gRead[int2(c.x, max(c.y - 1, 0))].x, hS = gRead[int2(c.x, min(c.y + 1, N - 1))].x;
    const float hW = gRead[int2(max(c.x - 1, 0), c.y)].x, hE = gRead[int2(min(c.x + 1, N - 1), c.y)].x;
    float h = ((hN + hS + hE + hW) * 0.5f - s.y) * gDamping;
    // Sponge border: extra damping over gSponge cells, so waves leave the region instead of reflecting.
    const float edge = float(min(min(c.x, c.y), min(N - 1 - c.x, N - 1 - c.y)));
    if (gSponge > 0.f) h *= lerp(0.9f, 1.f, saturate(edge / gSponge));

    const float2 cc = float2(c) + 0.5f;
    const float t = float(gStep) * gStepTime;
    const float cs = cellSize();
    for (uint k = 0; k < gDrops; k++)
    {
        const float2 dc = float2(rnd(gStep, k, 1u), rnd(gStep, k, 2u)) * float(N);
        h -= gDropStrength * bump(length(cc - dc), gDropRadius / cs);
    }
    for (uint e = 0; e < gEmitterCount; e++)
    {
        const float4 A = gEmitterA[e], B = gEmitterB[e];
        const float osc = sin(6.2831853f * (t / max(B.x, 1e-4f) + B.y));
        h += A.w * osc * bump(length(cc - toCell(A.xy)), A.z / cs);
    }
    for (uint m = 0; m < gMoverCount; m++)
    {
        const float4 A = gMoverA[m], B = gMoverB[m];
        if (t < B.x || t > B.y || B.y <= B.x) continue;
        const float2 q = lerp(A.xy, A.zw, (t - B.x) / (B.y - B.x));
        const float speed = length(A.zw - A.xy) / (B.y - B.x);
        h -= B.w * speed * gStepTime * bump(length(cc - toCell(q)), B.z / cs);
    }
    if (gApplyContact != 0 && gContact[c.y * N + c.x].load() > 0u)
        h += gContactStrength * sin(6.2831853f * t / max(gContactPeriod, 1e-4f));
    gWrite[c] = float4(h, s.x, 0.f, 1.f);
}

float fieldAt(int2 c)
{
    const int N = int(gGrid);
    return gField[clamp(c, int2(0), int2(N - 1))].x;
}

/** Bilinear height at continuous cell coordinates. */
float heightAt(float2 x)
{
    const float2 g = x - 0.5f;
    const float2 f = g - floor(g);
    const int2 i = int2(floor(g));
    const float a = fieldAt(i), b = fieldAt(i + int2(1, 0)), c = fieldAt(i + int2(0, 1)), d = fieldAt(i + int2(1, 1));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

[numthreads(16, 16, 1)]
void apply(uint3 tid: SV_DispatchThreadID)
{
    const uint2 p = tid.xy;
    if (any(p >= gFrameDim)) return;
    const float4 nw = gNormW[p];
    const float4 x = gPosW[p];
    gHeightOut[p] = float4(0.f, 0.f, 0.f, 1.f);
    if (gStrength == 0.f || x.w == 0.f || dot(nw.xyz, nw.xyz) == 0.f)
    {
        gNormWOut[p] = nw;
        return;
    }
    const float3 N = normalize(nw.xyz);
    float wt = (dot(N, gUp) >= gUpMin && abs(dot(x.xyz, gUp) - gLevel) <= gPlaneTolerance) ? 1.f : 0.f;
    const float2 c = toCell(float2(dot(x.xyz, gT1), dot(x.xyz, gT2)));
    const float Nf = float(gGrid);
    const float edge = min(min(c.x, c.y), min(Nf - c.x, Nf - c.y));
    wt *= edge <= 0.f ? 0.f : (gEdgeFade > 0.f ? saturate(edge / (gEdgeFade * Nf)) : 1.f);
    if (gHasMask != 0) wt *= saturate(gMask[p][gMaskChannel]);
    if (wt == 0.f)
    {
        gNormWOut[p] = nw;
        return;
    }
    const float h = heightAt(c);
    const float cs = cellSize();
    const float gx = (heightAt(c + float2(1.f, 0.f)) - heightAt(c - float2(1.f, 0.f))) / (2.f * cs);
    const float gy = (heightAt(c + float2(0.f, 1.f)) - heightAt(c - float2(0.f, 1.f))) / (2.f * cs);
    gHeightOut[p] = float4(h, wt, gx, gy);
    if (gx == 0.f && gy == 0.f)
    {
        gNormWOut[p] = nw;
        return;
    }
    gNormWOut[p] = float4(normalize(N - gStrength * wt * (gx * gT1 + gy * gT2)), nw.w);
}
