/**
 * WebFalcor override of RenderPasses/RTXDIPass/LoadShadingData.slang
 * (docs §4.3). Diff vs upstream: `sd = {};` fails in newer Slang for
 * WGSL (ShadingData has explicit initialization) — replaced with the
 * prepared-default pattern used across the other pass overrides.
 */
__exported import Scene.Scene;
__exported import Scene.Shading;
__exported import Scene.HitInfo;
import Scene.Material.ShadingUtils;
import Utils.Math.MathHelpers;

/**
 * Determine hints to use when creating the material instance.
 */
uint getMaterialInstanceHints()
{
    uint hints = 0;
#if GBUFFER_ADJUST_SHADING_NORMALS
    hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
#endif
    return hints;
}

/**
 * Helper for setting up the ShadingData struct based on loaded data.
 * @param[in] pixel Current pixel coordinates.
 * @param[in] frameDim Frame dimensions in pixel.
 * @param[in] camera Current camera.
 * @param[in] vbuffer VBuffer texture.
 * @param[out] sd ShadingData struct.
 * @return True if the pixel has valid data (not a background pixel). Note sd.V is always valid.
 */
bool loadShadingData(const uint2 pixel, const uint2 frameDim, const Camera camera, Texture2D<PackedHitInfo> vbuffer, out ShadingData sd)
{
    // Newer Slang rejects `sd = {};` (explicit-init struct); members are
    // fully assigned below for valid hits and sd.V for misses.
    ShadingData defaultSd;
    sd = defaultSd;

    const float3 rayDir = camera.computeRayPinhole(pixel, frameDim).dir;
    bool valid = false;

    const HitInfo hit = HitInfo(vbuffer[pixel]);
    if (hit.isValid() && hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        const VertexData v = gScene.getVertexData(triangleHit);
        const uint materialID = gScene.getMaterialID(triangleHit.instanceID);
        sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

        valid = true;
    }

    sd.V = -rayDir;
    return valid;
}
