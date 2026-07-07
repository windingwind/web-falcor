// Reference for SVO location-code primitives (SDFVoxelCommon.slang).
#include <cstdint>
#include <cstdio>
static const uint32_t kMaxLevel = 19;
static const uint32_t kLevelOffset = 3 * kMaxLevel;
static const uint64_t kCoordsMask = (1ull << (3 * kMaxLevel)) - 1;
static uint64_t shiftCoord(uint32_t x) {
    uint64_t y = x;
    y = (y | y << 32) & 0x1f00000000ffffull;
    y = (y | y << 16) & 0x1f0000ff0000ffull;
    y = (y | y << 8) & 0x100f00f00f00f00full;
    y = (y | y << 4) & 0x10c30c30c30c30c3ull;
    y = (y | y << 2) & 0x1249249249249249ull;
    return y;
}
static uint64_t encodeLocation(uint32_t x, uint32_t y, uint32_t z, uint32_t level) {
    auto g = [&](uint32_t c){ return (uint64_t)((c & ((1u<<kMaxLevel)-1)) << (kMaxLevel - level)); };
    uint64_t sx = shiftCoord(g(x)), sy = shiftCoord(g(y)), sz = shiftCoord(g(z));
    return ((uint64_t)level << kLevelOffset) | (((sx << 2) | (sy << 1) | sz) & kCoordsMask);
}
static uint64_t createChild(uint64_t code, uint32_t childID) {
    uint32_t level = 1 + ((code >> kLevelOffset) & 0x1f);
    uint32_t lvl = level < kMaxLevel ? level : kMaxLevel;
    uint64_t bits = code & kCoordsMask;
    bits |= (uint64_t)(childID & 0x7) << (kLevelOffset - 3 * lvl);
    bits &= kCoordsMask;
    bits |= ((uint64_t)lvl << kLevelOffset);
    return bits;
}
int main() {
    printf("{\"enc\":[");
    uint32_t probes[5][4] = {{0,0,0,0},{1,2,3,7},{5,5,5,3},{127,0,63,7},{1,0,0,1}};
    for (int i=0;i<5;i++){ printf("%s\"%llu\"", i?",":"", (unsigned long long)encodeLocation(probes[i][0],probes[i][1],probes[i][2],probes[i][3])); }
    printf("],\"child\":[");
    uint64_t root = encodeLocation(0,0,0,0);
    for (uint32_t c=0;c<8;c++){ printf("%s\"%llu\"", c?",":"", (unsigned long long)createChild(root,c)); }
    printf("]}\n");
    return 0;
}
