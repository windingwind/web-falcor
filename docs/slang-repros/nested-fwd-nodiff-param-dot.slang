// Crashes slangc 2026.18.2 (see docs/module-mapping.md §6.9): slangc THIS.slang -target wgsl -entry main -stage compute
RWStructuredBuffer<float> o;
struct Q : IDifferentiable { float s; uint2 pixel; };
struct Gen { uint s; };
[Differentiable]
[PreferRecompute]
__generic<let N : int> float3[N] g(Q sceneQuery, no_diff float kappa, float3 origin, no_diff float3 baseDirection, float3 direction[N], inout Gen sg)
{
    float3 warpedDirection[N];
    float totalWeight[N];
    float3 totalWarpedDirection[N];
    [ForceUnroll]
    for (int j = 0; j < N; j++) { totalWeight[j] = 0.f; totalWarpedDirection[j] = float3(0.f); }
    [ForceUnroll]
    for (int j = 0; j < N; j++) { totalWeight[j] = 1.f + dot(direction[j], baseDirection); }
    [ForceUnroll]
    for (int j = 0; j < N; j++) { totalWeight[j] = max(totalWeight[j], 1e-8f); warpedDirection[j] = totalWarpedDirection[j] / float3(totalWeight[j]) + direction[j]; }
    return warpedDirection;
}
[Differentiable]
[PreferRecompute]
float3[3] f(Q q, no_diff float kappa, float3 origin, float3 base, float3 d[3], inout Gen sg)
{
    float3 m[3] = g<3>(q, kappa, origin, base, d, sg);
    float3 w[3];
    [ForceUnroll] for (int j = 0; j < 3; j++) w[j] = normalize(m[j]);
    return w;
}
[Differentiable]
[PreferRecompute]
float h(Q q, float3 x, inout Gen sg)
{
    DifferentialPair<float3[3]> dp = DifferentialPair<float3[3]>({ x, x, x }, { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1) });
    Q.Differential dq; dq.s = 0.f;
    Gen sgCopy = sg;
    DifferentialPair<float3[3]> r = fwd_diff(f)(diffPair(q, dq), 2.f, diffPair(x, float3(0)), diffPair(x, float3(0)), dp, sgCopy);
    float3 wd = r.p[0] - detach(r.p[0]);
    float3x3 J = no_diff float3x3(1,0,0,0,1,0,0,0,1) + float3x3(r.d[0] - detach(r.d[0]), r.d[1] - detach(r.d[1]), r.d[2] - detach(r.d[2]));
    return determinant(J) + wd.x;
}
[numthreads(1,1,1)]
void main() { Q q; q.s = 1.f; q.pixel = uint2(0); Gen sg = { 1 }; var dq = diffPair(q); var dp = diffPair(float3(o[0], o[1], o[2])); bwd_diff(h)(dq, dp, sg, 1.f); o[3] = dp.d.x; }
