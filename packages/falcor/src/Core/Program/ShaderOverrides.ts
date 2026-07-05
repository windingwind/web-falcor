/**
 * WebFalcor shader overrides (DESIGN.md §4.3).
 *
 * Upstream shader files that use features WGSL cannot express are substituted
 * with web-owned implementations that keep the exact interface (entry points,
 * defines, cbuffers, binding names). Host code keeps referencing the upstream
 * path; the ProgramManager resolves content through this map. Each override
 * file documents its diff vs upstream.
 */

export const kShaderOverrides: Readonly<Record<string, string>> = {
    // RWBuffer texel buffers + warp-size-32 wave reduction -> structured buffers + portable shared-memory reduction.
    "Utils/Algorithm/ParallelReduction.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/ParallelReduction.cs.slang",
    // ByteAddressBuffer atomics -> structured buffers with Atomic<uint> elements.
    "Utils/Algorithm/PrefixSum.cs.slang": "WebFalcor/Overrides/Utils/Algorithm/PrefixSum.cs.slang",
};
