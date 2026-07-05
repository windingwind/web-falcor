/** Matrix cbuffer layout test kernel (see program-system.gpu.test.ts). */

cbuffer CB
{
    float4x4 gWorld;
    float4 gPoint;
}

RWStructuredBuffer<float4> gOut;

[numthreads(1, 1, 1)]
void main()
{
    // Column-vector transform, Falcor convention.
    gOut[0] = mul(gWorld, gPoint);
}
