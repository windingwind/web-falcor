/** Evaluates the 16 real SH basis functions (l <= 3) at gDirs[i]; validates Utils/Math/SphericalHarmonics on WGSL. */
import Utils.Math.SphericalHarmonics;

StructuredBuffer<float4> gDirs;
RWStructuredBuffer<float> gResult; // 16 values per direction

[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= (uint)gDirs.getCount()) return;
    float3 p = gDirs[i].xyz;
    for (int l = 0; l <= 3; l++)
        for (int m = -l; m <= l; m++)
            gResult[i * 16 + get_SH_index(l, m)] = eval_SH(l, m, p);
}
