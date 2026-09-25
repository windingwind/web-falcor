/**
 * Mirrors Falcor/Core/API/Raytracing.h. WebGPU has no hardware ray tracing; the flags keep native's
 * values (the RAY_FLAG_* shader constants) for the software ray queries.
 */

export enum RayFlags {
    None = 0,
    ForceOpaque = 0x1,
    ForceNonOpaque = 0x2,
    AcceptFirstHitAndEndSearch = 0x4,
    SkipClosestHitShader = 0x8,
    CullBackFacingTriangles = 0x10,
    CullFrontFacingTriangles = 0x20,
    CullOpaque = 0x40,
    CullNonOpaque = 0x80,
    SkipTriangles = 0x100,
    SkipProceduralPrimitives = 0x200,
}

/** Maximum raytracing attribute size. */
export function getRaytracingMaxAttributeSize(): number {
    return 32;
}
