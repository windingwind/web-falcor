/**
 * Sanity-check kernel: exercises the upstream Falcor shader library
 * (Utils.Math, Utils.Sampling) through the Slang->WGSL pipeline.
 */
import Utils.Math.MathHelpers;
import Utils.Sampling.TinyUniformSampleGenerator;

RWStructuredBuffer<float4> gOutput;

cbuffer PerFrame
{
    uint gFrameIndex;
    uint gElementCount;
}

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gElementCount) return;
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(uint2(tid.x, 0), gFrameIndex);
    float3 dir = sample_sphere(sampleNext2D(sg));
    gOutput[tid.x] = float4(dir, 1.0);
}
