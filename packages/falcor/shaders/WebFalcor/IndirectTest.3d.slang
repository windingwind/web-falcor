// Fullscreen-triangle raster kernel for the indirect-draw tests.
struct VSOut
{
    float4 posH : SV_POSITION;
};

VSOut vsMain(uint id: SV_VertexID)
{
    VSOut o;
    float2 uv = float2((id << 1) & 2, id & 2);
    o.posH = float4(uv * 2.f - 1.f, 0.f, 1.f);
    return o;
}

float4 psMain() : SV_TARGET0
{
    return float4(1.f, 0.f, 0.f, 1.f);
}
