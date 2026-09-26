// Reference for std-algorithms.test.ts: g++ -O2 std-sort-reference.cpp && ./a.out > std-sort-reference.txt (libstdc++).
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdint>
// Deterministic inputs: an LCG, values with many ties; prints the sorted index permutation.
int main() {
    uint32_t s = 12345;
    auto next = [&]() { s = s * 1664525u + 1013904223u; return s >> 8; };
    for (int n : {5, 17, 40, 100, 1000, 3000}) {
        std::vector<float> w(n);
        for (auto& x : w) x = float(next() % 7);
        std::vector<uint32_t> p(n);
        for (int i = 0; i < n; i++) p[i] = i;
        std::sort(p.begin(), p.end(), [&](uint32_t a, uint32_t b) { return w[a] < w[b]; });
        printf("%d:", n);
        for (auto v : p) printf("%u,", v);
        printf("\n");
    }
    // std::nth_element at a few positions over the same kind of input.
    for (int n : {9, 50, 400, 2500}) {
        std::vector<float> w(n);
        for (auto& x : w) x = float(next() % 5);
        for (int k : {0, n / 3, n / 2, n - 1}) {
            std::vector<uint32_t> p(n);
            for (int i = 0; i < n; i++) p[i] = i;
            std::nth_element(p.begin(), p.begin() + k, p.end(), [&](uint32_t a, uint32_t b) { return w[a] < w[b]; });
            printf("nth %d %d:", n, k);
            for (auto v : p) printf("%u,", v);
            printf("\n");
        }
    }
}
