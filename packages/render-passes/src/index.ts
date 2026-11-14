/**
 * @web-falcor/render-passes — render pass plugin registry.
 * One directory per pass, mirroring Falcor/Source/RenderPasses (see docs §RenderPasses
 * for the full parity list). Importing this module registers all passes with
 * the render-graph pass factory (replaces DLL plugin loading).
 */

export * from "./AccumulatePass/AccumulatePass.js";
export * from "./ToneMapper/ToneMapper.js";
export * from "./BlitPass/BlitPass.js";
export * from "./ImageLoader/ImageLoader.js";
export * from "./GBuffer/GBufferRaster.js";
export * from "./GBuffer/VBufferRT.js";
export * from "./GBuffer/VBufferRaster.js";
export * from "./RenderPassTemplate.js";
export * from "./OverlaySamplePass.js";
export * from "./GBuffer/GBufferRT.js";
export * from "./MinimalPathTracer/MinimalPathTracer.js";
export * from "./PathTracer/PathTracer.js";
export * from "./Utils/Composite.js";
export * from "./Utils/CrossFade.js";
export * from "./Utils/GaussianBlur.js";
export * from "./DebugPasses/ComparisonPass.js";
export * from "./DebugPasses/ColorMapPass.js";
export * from "./DebugPasses/InvalidPixelDetectionPass.js";
export * from "./ErrorMeasurePass/ErrorMeasurePass.js";
export * from "./Utils/ModulateIllumination.js";
export * from "./Utils/SimplePostFX.js";
export * from "./Utils/FLIPPass.js";
export * from "./WhittedRayTracer/WhittedRayTracer.js";
export * from "./SceneDebugger/SceneDebugger.js";
export * from "./PixelInspectorPass/PixelInspectorPass.js";
export * from "./TAA/TAA.js";
export * from "./SVGFPass/SVGFPass.js";
export * from "./BSDFViewer/BSDFViewer.js";
export * from "./RTXDIPass/RTXDIPass.js";
export * from "./TestPasses/TestRtProgram.js";
export * from "./WARDiffPathTracer/WARDiffPathTracer.js";
export * from "./SDFEditor/SDFEditor.js";
export * from "./BSDFOptimizer/BSDFOptimizer.js";
export * from "./NRDPass/NRDPass.js";
export * from "./RasterLighting/CascadedShadowMap.js";
export * from "./RasterLighting/DeferredLighting.js";
export * from "./RasterLighting/VisibilityService.js";
export * from "./RasterLighting/VisibilityQuery.js";
export * from "./SSAO/SSAO.js";
export * from "./PhysicalSky/PhysicalSky.js";
export * from "./FXAA/FXAA.js";
export * from "./DepthOfField/DepthOfField.js";
export * from "./MotionBlur/MotionBlur.js";
export * from "./MotionStats/MotionStats.js";
export * from "./Stylize/StyleAovs.js";
export * from "./Stylize/RimLight.js";
export * from "./Stylize/RampShade.js";
export * from "./Utils/History.js";
export * from "./Utils/TemporalBlend.js";
export * from "./Utils/HiZBuild.js";
export * from "./Layers/DepthComposite.js";
export * from "./Layers/ParticleLayer.js";
export * from "./FroxelFog/FroxelFog.js";
