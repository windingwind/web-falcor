/** HoldFrames (see HoldFrames.ts): copy gIn -> gOut (used for src -> history on new frames and history -> dst on held frames). */
cbuffer CB { uint2 gFrameDim; };
Texture2D<float4> gIn;
RWTexture2D<float4> gOut;
[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    if (any(tid.xy >= gFrameDim)) return;
    gOut[tid.xy] = gIn[tid.xy];
}
