// BrickedGrid probe: per index coordinate, the NanoVDB value, the brick lookup and the smallest
// (majorant - value) over the range texture's 4 mips.
import Scene.Scene;

cbuffer CB
{
    uint gCount;
    uint gGridIndex;
}

StructuredBuffer<int4> gCoords;
RWStructuredBuffer<float4> gResults;

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gCount)
        return;
    Grid grid;
    gScene.getGrid(gGridIndex, grid);
    Grid::Accessor accessor = grid.createAccessor();
    const int3 c = gCoords[tid.x].xyz;
    const float v = grid.lookupIndex(c, accessor);
    const int3 rel = c - grid.minIndex;
    float slack = 1e30f;
    for (int mip = 0; mip < 4; mip++) slack = min(slack, grid.lookupIndexLocalMajorantTex(rel, mip) - v);
    gResults[tid.x] = float4(v, grid.lookupIndexTex(rel), slack, 0.f);
}
