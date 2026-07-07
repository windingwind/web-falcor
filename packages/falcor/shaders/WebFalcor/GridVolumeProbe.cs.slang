// Dumps gScene grid-volume state for host-side verification.
import Scene.Scene;

RWStructuredBuffer<float> gOut;

[numthreads(1, 1, 1)]
void main()
{
    gOut[0] = float(gScene.getGridVolumeCount());
    GridVolume v = gScene.getGridVolume(0);
    gOut[1] = float(v.data.densityGrid);
    gOut[2] = v.data.densityScale;
    gOut[3] = v.data.boundsMin.x;
    gOut[4] = v.data.boundsMin.y;
    gOut[5] = v.data.boundsMin.z;
    gOut[6] = v.data.boundsMax.x;
    gOut[7] = v.data.boundsMax.y;
    gOut[8] = v.data.boundsMax.z;
    gOut[9] = v.data.albedo.x;
    gOut[10] = float(v.hasDensityGrid() ? 1.0 : 0.0);
    Grid grid;
    gScene.getGrid(0, grid);
    Grid::Accessor acc = grid.createAccessor();
    gOut[11] = grid.lookupIndex(int3(56, 112, 56), acc);

    // Replicates SceneDebugger::evalGridVolumeTransmittance for the camera
    // center ray of the smoke scene (pos 25,0,55 -> target 0,25,0).
    const float3 pos = float3(25.f, 0.f, 55.f);
    const float3 dir = normalize(float3(-25.f, 25.f, -55.f));
    const float3 bmin = v.data.boundsMin;
    const float3 bmax = v.data.boundsMax;
    const float3 invDir = 1.f / dir;
    const float3 t0 = (bmin - pos) * invDir;
    const float3 t1 = (bmax - pos) * invDir;
    float2 nearFar = float2(max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z)),
                            min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z)));
    nearFar.x = max(nearFar.x, 0.f);
    gOut[12] = nearFar.x;
    gOut[13] = nearFar.y;
    float opticalDepth = 0.f;
    Grid densityGrid;
    gScene.getGrid(v.data.densityGrid, densityGrid);
    Grid::Accessor acc2 = densityGrid.createAccessor();
    const float3 ipos = mul(v.data.invTransform, float4(pos, 1.f)).xyz;
    const float3 idir = mul(v.data.invTransform, float4(dir, 0.f)).xyz;
    [loop]
    for (uint step = 0; step < 500; ++step)
    {
        float t = lerp(nearFar.x, nearFar.y, (step + 0.5f) / 500);
        opticalDepth += densityGrid.lookupIndex(int3(ipos + t * idir), acc2);
    }
    opticalDepth *= (nearFar.y - nearFar.x) / 500 * v.data.densityScale;
    gOut[14] = exp(-opticalDepth);
    gOut[15] = ipos.x;
    gOut[16] = ipos.y;
    gOut[17] = ipos.z;
    gOut[18] = v.data.invTransform[0][0];
    gOut[19] = v.data.invTransform[0][3];
    gOut[20] = v.data.invTransform[3][0];
    gOut[21] = v.data.invTransform[1][3];
    float3 mid = ipos + lerp(nearFar.x, nearFar.y, 0.5f) * idir;
    gOut[22] = densityGrid.lookupIndex(int3(mid), acc2);
}
