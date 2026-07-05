/**
 * @web-falcor/render-passes — render pass plugin registry.
 * One directory per pass, mirroring Falcor/Source/RenderPasses (see DESIGN.md §RenderPasses
 * for the full parity list). Importing this module registers all passes with
 * the render-graph pass factory (replaces DLL plugin loading).
 */

export * from "./AccumulatePass/AccumulatePass.js";
export * from "./ToneMapper/ToneMapper.js";
export * from "./BlitPass/BlitPass.js";
