/**
 * WebFalcor DeferredLighting helper: projects the scene env map (radiance) onto
 * order-2 spherical harmonics (9 coefficients) for diffuse irradiance.
 * One workgroup of 64 threads integrates a kThetaRes x kPhiRes direction grid
 * (cell-centre midpoint rule with exact cell solid angles) and reduces in shared memory.
 * Uses EnvMap.eval(dir) so the env-map rotation/intensity handling is the scene's.
 */
#include "Utils/Math/MathConstants.slangh"
import Scene.Shading;

RWStructuredBuffer<float4> gSH; ///< 9 entries: rgb = SH radiance coefficient L_lm.

static const uint kThreads = 64;
static const uint kThetaRes = 128;
static const uint kPhiRes = 256;

groupshared float3 sAcc[kThreads * 9];

void shBasis(float3 d, out float Y[9])
{
    Y[0] = 0.282095f;
    Y[1] = 0.488603f * d.y;
    Y[2] = 0.488603f * d.z;
    Y[3] = 0.488603f * d.x;
    Y[4] = 1.092548f * d.x * d.y;
    Y[5] = 1.092548f * d.y * d.z;
    Y[6] = 0.315392f * (3.f * d.z * d.z - 1.f);
    Y[7] = 1.092548f * d.x * d.z;
    Y[8] = 0.546274f * (d.x * d.x - d.y * d.y);
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_GroupThreadID)
{
    float3 acc[9];
    for (uint k = 0; k < 9; k++) acc[k] = float3(0.f);
    const float dPhi = 2.f * M_PI / kPhiRes;
    for (uint it = tid.x; it < kThetaRes; it += kThreads)
    {
        const float t0 = M_PI * it / kThetaRes;
        const float t1 = M_PI * (it + 1) / kThetaRes;
        const float theta = 0.5f * (t0 + t1);
        const float dOmega = dPhi * (cos(t0) - cos(t1)); // exact solid angle of the cell
        for (uint ip = 0; ip < kPhiRes; ip++)
        {
            const float phi = (ip + 0.5f) * dPhi;
            const float3 d = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            const float3 L = gScene.envMap.eval(d);
            float Y[9];
            shBasis(d, Y);
            for (uint k = 0; k < 9; k++) acc[k] += L * (Y[k] * dOmega);
        }
    }
    for (uint k = 0; k < 9; k++) sAcc[tid.x * 9 + k] = acc[k];
    GroupMemoryBarrierWithGroupSync();
    if (tid.x < 9)
    {
        float3 s = float3(0.f);
        for (uint t = 0; t < kThreads; t++) s += sAcc[t * 9 + tid.x];
        gSH[tid.x] = float4(s, 0.f);
    }
}
