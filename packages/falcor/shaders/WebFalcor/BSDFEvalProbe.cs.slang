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
    sd.V = normalize(gDirections[tid.x * 2].xyz);
    float3 wo = normalize(gDirections[tid.x * 2 + 1].xyz);

    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);
    UniformSampleGenerator sg = UniformSampleGenerator(tid.xy, 0);
    float3 f = mi.eval(sd, wo, sg);
    // Also report the reported albedo so the LUT path is observable.
    gResults[tid.x] = float4(f, mi.getProperties(sd).diffuseReflectionAlbedo.g);
}
