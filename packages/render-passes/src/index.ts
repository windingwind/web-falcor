/**
 * @web-falcor/render-passes — render pass plugin registry.
 * One directory per pass, mirroring Falcor/Source/RenderPasses (see DESIGN.md §RenderPasses
 * for the full parity list). Importing this module registers all passes with
 * the render-graph pass factory (replaces DLL plugin loading).
 */

export * from "./AccumulatePass/AccumulatePass.js";
export * from "./ToneMapper/ToneMapper.js";
export * from "./BlitPass/BlitPass.js";
export * from "./ImageLoader/ImageLoader.js";
export * from "./GBuffer/GBufferRaster.js";
export * from "./GBuffer/VBufferRT.js";
export * from "./GBuffer/GBufferRT.js";
export * from "./MinimalPathTracer/MinimalPathTracer.js";
export * from "./PathTracer/PathTracer.js";
export * from "./Utils/Composite.js";
export * from "./Utils/CrossFade.js";
export * from "./Utils/GaussianBlur.js";
export * from "./DebugPasses/ComparisonPass.js";
export * from "./DebugPasses/ColorMapPass.js";
export * from "./Utils/ModulateIllumination.js";
export * from "./Utils/SimplePostFX.js";
export * from "./Utils/FLIPPass.js";
export * from "./WhittedRayTracer/WhittedRayTracer.js";
export * from "./SceneDebugger/SceneDebugger.js";
export * from "./BSDFViewer/BSDFViewer.js";
