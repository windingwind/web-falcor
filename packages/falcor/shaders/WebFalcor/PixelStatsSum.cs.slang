// Sums each region of the packed pixel-stats buffer (region layout in the
// PixelStats override) into gResult[region] for the CPU getStats() aggregate
// (native reduces its per-type stats textures with ParallelReduction instead).

cbuffer SumCB
{
    uint gNumElems; // elements per region (frameDim.x * frameDim.y)
}

StructuredBuffer<uint> gStatsBuffer;

RWStructuredBuffer<Atomic<uint>> gResult; // one total per region

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (dispatchThreadId.x >= gNumElems)
        return;
    const uint v = gStatsBuffer[dispatchThreadId.y * gNumElems + dispatchThreadId.x];
    if (v != 0)
        gResult[dispatchThreadId.y].add(v);
}
