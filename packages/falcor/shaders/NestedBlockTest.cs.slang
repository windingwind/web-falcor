/** Nested ParameterBlock test (gScene binding pattern; see program-system.gpu.test.ts). */

struct InnerBuffers
{
    StructuredBuffer<float4> data0;
};

struct TestBlock
{
    float4x4 transform;
    float3 offset;
    uint count;
    InnerBuffers vertices;
    Texture2D colorTex;
    SamplerState sampler;
};

ParameterBlock<TestBlock> gBlock;

RWStructuredBuffer<float4> gOut;

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gBlock.count) return;
    float4 v = gBlock.vertices.data0[tid.x];
    float4 transformed = mul(gBlock.transform, v);
    float4 texel = gBlock.colorTex.SampleLevel(gBlock.sampler, float2(0.5, 0.5), 0.0);
    gOut[tid.x] = transformed + float4(gBlock.offset, 0.0) + texel;
}
