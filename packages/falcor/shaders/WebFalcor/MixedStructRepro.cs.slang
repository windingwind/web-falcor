// Minimal repro for the SceneDebugger binding shape: a cbuffer holding one
// struct that mixes a nested uniform struct with resource members.
struct Params
{
    uint mode;
    uint2 frameDim;
    uint frameCount;
};

struct Mixed
{
    Params params;
    StructuredBuffer<uint> data;
    WTexture2D<float4> output;
    RWStructuredBuffer<uint> result;

    void run(uint2 pixel)
    {
        if (any(pixel >= params.frameDim))
            return;
        if (all(pixel == uint2(0, 0)))
        {
            result[0] = params.mode;
            result[1] = params.frameDim.x;
            result[2] = params.frameDim.y;
            result[3] = params.frameCount + data[0];
        }
        output.Store(pixel, float4(1.0));
    }
};

cbuffer CB
{
    Mixed gMixed;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gMixed.run(dispatchThreadId.xy);
}
