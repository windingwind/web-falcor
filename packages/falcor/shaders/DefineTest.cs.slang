/** Define-variant test kernel (see program-system.gpu.test.ts). */

RWStructuredBuffer<uint> gData;

#ifndef SCALE_MODE
#define SCALE_MODE 1
#endif

[numthreads(32, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
#if SCALE_MODE == 2
    gData[tid.x] = tid.x * 100;
#else
    gData[tid.x] = tid.x;
#endif
}
