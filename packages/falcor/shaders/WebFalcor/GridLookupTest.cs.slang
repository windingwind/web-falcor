// Grid lookup probe: traverses the scene's NanoVDB grid (gScene.grid0) via
// PNanoVDB read accessors at given index coordinates - validates the whole
// WGSL traversal chain over the web-built grid buffer.
import Scene.Scene;

cbuffer CB
{
    uint gCount;
}

StructuredBuffer<int4> gCoords;
RWStructuredBuffer<float> gResults;

[numthreads(64, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= gCount)
        return;
    Grid grid;
    gScene.getGrid(0, grid);
    Grid::Accessor accessor = grid.createAccessor();
    gResults[tid.x] = grid.lookupIndex(gCoords[tid.x].xyz, accessor);
}
