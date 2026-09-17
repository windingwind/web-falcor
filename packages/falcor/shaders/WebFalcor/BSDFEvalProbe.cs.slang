// BSDF evaluation probe: evaluates a scene material's BSDF at explicit local
// incident/outgoing direction pairs, on a canonical identity shading frame.
// Used to compare measured-BRDF materials against their analytic ground truth.
import Scene.Shading;
import Utils.Sampling.UniformSampleGenerator;

cbuffer CB
{
    uint gCount;
    uint gMaterialID;
}

/// wi in .xyz of even entries, wo in .xyz of odd entries (local frame, z up).
/// The two .w slots carry the shading point's uv, which materials that vary over
/// the surface (MERLMix's index map) need in order to pick their BRDF.
StructuredBuffer<float4> gDirections;
/// f(wi, wo) * wo.z per pair (what IMaterialInstance::eval returns).
RWStructuredBuffer<float4> gResults;

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gCount) return;

    ShadingData sd;
    sd.frame = ShadingFrame::createIdentity();
    sd.faceN = sd.frame.N;
    sd.frontFacing = true;
    sd.mtl = gScene.materials.getMaterialHeader(gMaterialID);
    sd.materialID = gMaterialID;
    sd.IoR = 1.f;
    sd.uv = float2(gDirections[tid.x * 2].w, gDirections[tid.x * 2 + 1].w);
    sd.V = normalize(gDirections[tid.x * 2].xyz);
    float3 wo = normalize(gDirections[tid.x * 2 + 1].xyz);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    UniformSampleGenerator sg = UniformSampleGenerator(tid.xy, 0);
    float3 f = mi.eval(sd, wo, sg);
    // Also report the reported albedo so the LUT path is observable.
    gResults[tid.x] = float4(f, mi.getProperties(sd).diffuseReflectionAlbedo.g);
}

/// Importance-sampling probe: draws samples from the material's own sampling
/// routine and accumulates the returned weights, whose mean estimates the
/// directional albedo. Disagreement with a uniform-hemisphere integral means the
/// sampling distribution and the evaluation disagree.
cbuffer SampleCB
{
    uint gSampleCount;   ///< Samples per incident direction.
    uint gSampleMtlID;
    uint gDirCount;      ///< Number of incident directions.
}

StructuredBuffer<float4> gIncident;
RWStructuredBuffer<float4> gSampleSums;

[numthreads(64, 1, 1)]
void mainSample(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gDirCount) return;

    ShadingData sd;
    sd.frame = ShadingFrame::createIdentity();
    sd.faceN = sd.frame.N;
    sd.frontFacing = true;
    sd.mtl = gScene.materials.getMaterialHeader(gSampleMtlID);
    sd.materialID = gSampleMtlID;
    sd.IoR = 1.f;
    sd.uv = float2(gIncident[tid.x].w, 0.f);
    sd.V = normalize(gIncident[tid.x].xyz);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    UniformSampleGenerator sg = UniformSampleGenerator(uint2(tid.x, 0), 0);

    float3 sum = float3(0.f);
    uint valid = 0;
    for (uint i = 0; i < gSampleCount; i++)
    {
        BSDFSample result;
        if (mi.sample(sd, sg, result, true) && result.pdf > 0.f)
        {
            sum += result.weight;
            valid++;
        }
    }
    // Mean weight over all attempts: failed samples contribute zero, as in a path tracer.
    gSampleSums[tid.x] = float4(sum / max(gSampleCount, 1u), float(valid));
}
