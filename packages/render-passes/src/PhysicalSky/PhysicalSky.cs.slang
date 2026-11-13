/** PhysicalSky kernel (see PhysicalSky.ts; its evalPhysicalSky() is the CPU reference of this shader).
 *  Preetham et al. 1999: L(theta, gamma) = L_z * F(theta, gamma) / F(0, theta_s) for Y, x, y with the Perez function
 *  F = (1 + A exp(B / cos theta)) (1 + C exp(D gamma) + E cos^2 gamma); coefficients and zenith values come from
 *  the host (turbidity-dependent). xyY -> XYZ -> linear sRGB (D65). Up axis +y. */

cbuffer CB
{
    uint2 gFrameDim;
    uint gHasColor;
    uint gHasPos;
    float3 gCamRight;   // camera right * tan(fovY/2) * aspect
    float gThetaS;
    float3 gCamUp;      // camera up * tan(fovY/2)
    float gSkyScale;
    float3 gCamFwd;
    float gCosSunRadius;
    float3 gSunDir;     // towards the sun
    float _pad0;
    float4 gCoefY0;     // A..D for Y
    float4 gCoefX0;     // A..D for x
    float4 gCoefYy0;    // A..D for y
    float4 gCoefE;      // E for (Y, x, y)
    float3 gZenith;     // (Y_z [kcd/m^2], x_z, y_z)
    float _pad1;
    float3 gSunRadiance;
    float _pad2;
    float3 gGround;
    float _pad3;
}

Texture2D<float4> gColor;
Texture2D<float4> gPosW;
RWTexture2D<float4> gDst;

float perez(float4 c, float e, float cosTheta, float gamma, float cosGamma)
{
    return (1.f + c.x * exp(c.y / max(cosTheta, 1e-3f))) * (1.f + c.z * exp(c.w * gamma) + e * cosGamma * cosGamma);
}

float3 xyYToLinearSrgb(float x, float y, float Y)
{
    const float X = x * Y / y, Z = (1.f - x - y) * Y / y;
    return float3(3.2404542f * X - 1.5371385f * Y - 0.4985314f * Z, -0.9692660f * X + 1.8760108f * Y + 0.0415560f * Z,
                  0.0556434f * X - 0.2040259f * Y + 1.0572252f * Z);
}

float3 evalSky(float3 d)
{
    const bool below = d.y < 0.f;
    const float3 dd = normalize(float3(d.x, max(d.y, 0.f), d.z));
    const float cosTheta = dd.y;
    const float cosGamma = clamp(dot(dd, gSunDir), -1.f, 1.f);
    const float gamma = acos(cosGamma);
    const float cosTS = cos(gThetaS);
    const float fY = perez(gCoefY0, gCoefE.x, cosTheta, gamma, cosGamma) / perez(gCoefY0, gCoefE.x, 1.f, gThetaS, cosTS);
    const float fx = perez(gCoefX0, gCoefE.y, cosTheta, gamma, cosGamma) / perez(gCoefX0, gCoefE.y, 1.f, gThetaS, cosTS);
    const float fy = perez(gCoefYy0, gCoefE.z, cosTheta, gamma, cosGamma) / perez(gCoefYy0, gCoefE.z, 1.f, gThetaS, cosTS);
    float3 rgb = max(xyYToLinearSrgb(gZenith.y * fx, gZenith.z * fy, gZenith.x * fY), float3(0.f)) * gSkyScale;
    if (below) return rgb * gGround;
    if (dot(d, gSunDir) >= gCosSunRadius) rgb += gSunRadiance;
    return rgb;
}

[numthreads(16, 16, 1)]
void main(uint3 tid: SV_DispatchThreadID)
{
    const uint2 pix = tid.xy;
    if (any(pix >= gFrameDim)) return;
    if (gHasPos != 0 && gPosW[pix].w != 0.f)
    {
        gDst[pix] = gHasColor != 0 ? gColor[pix] : float4(0.f, 0.f, 0.f, 1.f);
        return;
    }
    const float2 ndc = (float2(pix) + 0.5f) / float2(gFrameDim) * 2.f - 1.f;
    const float3 d = normalize(gCamFwd + ndc.x * gCamRight - ndc.y * gCamUp);
    gDst[pix] = float4(evalSky(d), 1.f);
}
