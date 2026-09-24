// Crashes slangc 2026.18.2 (see docs/module-mapping.md §6.9): slangc THIS.slang -target wgsl -entry main -stage compute
RWStructuredBuffer<float> o;
[Differentiable]
float3 g(float3 d, out float3 n) { n = d * 2.f; return d * 3.f; }
[Differentiable]
float3 f(float3 d) { float3 n; float3 r = g(d, n); return r * n; }
[Differentiable]
float h(float3 x) { DifferentialPair<float3> r = fwd_diff(f)(diffPair(x, float3(1, 0, 0))); return r.d.x + r.p.y; }
[numthreads(1,1,1)]
void main() { var dp = diffPair(float3(o[0], o[1], o[2])); bwd_diff(h)(dp, 1.f); o[3] = dp.d.x; }
