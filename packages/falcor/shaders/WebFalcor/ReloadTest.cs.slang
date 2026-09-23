// Minimal kernel for the in-session shader reload test: it writes a constant
// that the test rewrites in the source before asking for a reload.
RWStructuredBuffer<float> gOutput;

static const float kReloadTestValue = 11.0;

[numthreads(1, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    gOutput[tid.x] = kReloadTestValue;
}
