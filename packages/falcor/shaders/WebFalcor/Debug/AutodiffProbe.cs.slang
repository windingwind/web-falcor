/**
 * Slang automatic-differentiation feasibility probe (docs M8). Proves
 * reverse- AND forward-mode autodiff compile to WGSL *and* run correctly on
 * the WebGPU device — the portability gate for WARDiffPathTracer
 * (differentiable rendering). f(x) = x^2*k + sin(x); df/dx = 2xk + cos(x).
 */
[Differentiable]
float f(float x, no_diff float k)
{
    return x * x * k + sin(x);
}

RWStructuredBuffer<float> result;

[numthreads(1, 1, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    var dp = diffPair(2.0f, 0.0f);
    bwd_diff(f)(dp, 3.0f, 1.0f);   // seed dL/dout = 1
    result[0] = dp.d;              // dL/dx
    result[1] = fwd_diff(f)(diffPair(2.0f, 1.0f), 3.0f).d;
}
