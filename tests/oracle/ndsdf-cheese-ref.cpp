// Reference dump of SDFGrid::generateCheeseValues(128, 0) +
// NDSDFGrid::setValuesInternal quantization, gcc/libstdc++ exactly as native.
#include <random>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <algorithm>

static float len3(float x, float y, float z) { return std::sqrt(x*x + y*y + z*z); }

int main() {
    const float kHalfCheeseExtent = 0.4f;
    const uint32_t kHoleCount = 32, gridWidth = 128, seed = 0;
    float holes[kHoleCount][4];
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (uint32_t s = 0; s < kHoleCount; s++) {
        // gcc evaluates float3(dist,dist,dist) ctor args RIGHT-TO-LEFT:
        // x gets the 3rd draw, y the 2nd, z the 1st (verified vs order-test).
        float d1 = dist(rng), d2 = dist(rng), d3 = dist(rng);
        holes[s][0] = 2.0f * kHalfCheeseExtent * d3 - kHalfCheeseExtent;
        holes[s][1] = 2.0f * kHalfCheeseExtent * d2 - kHalfCheeseExtent;
        holes[s][2] = 2.0f * kHalfCheeseExtent * d1 - kHalfCheeseExtent;
        holes[s][3] = dist(rng) * 0.2f + 0.01f;
    }
    printf("{\"holes\":[");
    for (uint32_t s = 0; s < kHoleCount; s++)
        printf("%s[%.9g,%.9g,%.9g,%.9g]", s ? "," : "", holes[s][0], holes[s][1], holes[s][2], holes[s][3]);
    printf("],\n");

    const uint32_t W = gridWidth + 1;
    std::vector<float> corner(W * W * W);
    for (uint32_t z = 0; z < W; z++) for (uint32_t y = 0; y < W; y++) for (uint32_t x = 0; x < W; x++) {
        float plx = float(x) / gridWidth - 0.5f, ply = float(y) / gridWidth - 0.5f, plz = float(z) / gridWidth - 0.5f;
        float dx = std::fabs(plx) - kHalfCheeseExtent, dy = std::fabs(ply) - kHalfCheeseExtent, dz = std::fabs(plz) - kHalfCheeseExtent;
        float outside = len3(std::max(dx, 0.0f), std::max(dy, 0.0f), std::max(dz, 0.0f));
        float inside = std::min(std::max(std::max(dx, dy), dz), 0.0f);
        float sd = outside + inside;
        for (uint32_t s = 0; s < kHoleCount; s++)
            sd = std::max(sd, -(len3(plx - holes[s][0], ply - holes[s][1], plz - holes[s][2]) - holes[s][3]));
        corner[x + W * (y + W * z)] = std::clamp(sd, -float(M_SQRT2 * 1.2247448713915890491), float(M_SQRT2 * 1.2247448713915890491)); // sqrt3
    }
    // NDSDF quantization (narrowBandThickness 2.5 like NDSDFGrid.pyscene).
    const float narrowBand = 2.5f;
    uint32_t lodCount = 0; { uint32_t v = gridWidth / 8; while (v >>= 1) lodCount++; lodCount += 1; }
    uint32_t coarsest = gridWidth >> (lodCount - 1);
    float coarseNorm = 0.5f * float(M_SQRT2 * 1.2247448713915890491) * narrowBand / coarsest;
    printf("\"lodCount\":%u,\"coarsestWidth\":%u,\"coarsestNorm\":%.9g,\n", lodCount, coarsest, coarseNorm);
    // Sample probes: corner values + quantized snorm8 per LOD at fixed positions.
    printf("\"cornerSamples\":[");
    uint32_t probes[6][3] = {{0,0,0},{64,64,64},{64,64,90},{32,80,64},{1,2,3},{128,128,128}};
    for (int i = 0; i < 6; i++)
        printf("%s%.9g", i ? "," : "", corner[probes[i][0] + W * (probes[i][1] + W * probes[i][2])]);
    printf("],\n\"quantSamples\":[");
    bool first = true;
    for (uint32_t lod = 0; lod < lodCount; lod++) {
        uint32_t lw = 1 + (coarsest << lod);
        float norm = coarseNorm / float(1 << lod);
        uint32_t stride = 1u << (lodCount - lod - 1);
        uint32_t qp[3][3] = {{0,0,0},{lw/2,lw/2,lw/2},{lw/3,lw/2,(2*lw)/3}};
        for (int i = 0; i < 3; i++) {
            uint32_t x = qp[i][0], y = qp[i][1], z = qp[i][2];
            float v = corner[stride * (x + W * (y + W * z))];
            float n = std::clamp(v / norm, -1.0f, 1.0f);
            float is = n * 127.0f;
            int8_t q = is >= 0.0f ? int8_t(is + 0.5f) : int8_t(is - 0.5f);
            printf("%s[%u,%u,%u,%u,%d]", first ? "" : ",", lod, x, y, z, int(q));
            first = false;
        }
    }
    printf("]}\n");
    return 0;
}
