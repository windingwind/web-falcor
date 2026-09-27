// GridVolumeSampler probe: per ray, the mean transmittance (evalTransmittance) and the fraction of
// sampled scattering distances (sampleDistance) over gSamples samples.
import Scene.Scene;
import Rendering.Volumes.GridVolumeSampler;
import Utils.Sampling.TinyUniformSampleGenerator;
import Utils.Math.Ray;

cbuffer CB
{
    uint gRayCount;
    uint gSamples;
}

StructuredBuffer<float4> gRays; // origin, direction
RWStructuredBuffer<float2> gResults;

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gRayCount)
        return;
    const Ray ray = Ray(gRays[tid.x * 2].xyz, gRays[tid.x * 2 + 1].xyz, 0.f, 1e30f);
    GridVolumeSampler sampler = {};
    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(tid.x * 0x9e3779b9u + 17u);
    float tr = 0.f;
    float scattered = 0.f;
    for (uint s = 0; s < gSamples; s++)
    {
        tr += sampler.evalTransmittance(ray, sg);
        GridVolumeSampler::DistanceSample ds;
        if (sampler.sampleDistance(ray, sg, ds)) scattered += 1.f;
    }
    gResults[tid.x] = float2(tr, scattered) / gSamples;
}
