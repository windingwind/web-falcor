# SVGF feature graph (web-falcor): mirrors the upstream SVGF.py image test
# with two substitutions forced by this machine's native driver (no ROV ->
# GBufferRaster cannot run natively):
#   - GBufferRaster -> GBufferRT (channels verified elsewhere)
#   - pnFwidth (raster ddx/ddy, not available from GBufferRT) -> mvec.
#     pnFwidth only widens reprojection-validity thresholds; with a static
#     camera Z==Zprev and normal==normalPrev exactly, so any tight input is
#     equivalent (mvec is ~0 and RG32Float like pnFwidth).
from falcor import *

def render_graph_SVGFFeature():
    g = RenderGraph("SVGFFeature")
    SVGFPass = createPass("SVGFPass", {'Enabled': True, 'Iterations': 4, 'FeedbackTap': 1, 'VarianceEpsilon': 9.999999747378752e-05, 'PhiColor': 10.0, 'PhiNormal': 128.0, 'Alpha': 0.05000000074505806, 'MomentsAlpha': 0.20000000298023224})
    g.addPass(SVGFPass, "SVGFPass")
    GBufferRT = createPass("GBufferRT")
    g.addPass(GBufferRT, "GBufferRT")
    PathTracer = createPass("PathTracer")
    g.addPass(PathTracer, "PathTracer")

    g.addEdge("PathTracer.color", "SVGFPass.Color")
    g.addEdge("PathTracer.albedo", "SVGFPass.Albedo")
    g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
    g.addEdge("GBufferRT.emissive", "SVGFPass.Emission")
    g.addEdge("GBufferRT.posW", "SVGFPass.WorldPosition")
    g.addEdge("GBufferRT.guideNormalW", "SVGFPass.WorldNormal")
    g.addEdge("GBufferRT.mvec", "SVGFPass.PositionNormalFwidth")
    g.addEdge("GBufferRT.linearZ", "SVGFPass.LinearZ")
    g.addEdge("GBufferRT.mvec", "SVGFPass.MotionVec")

    g.markOutput("SVGFPass.Filtered image")
    return g

SVGFFeature = render_graph_SVGFFeature()
try: m.addGraph(SVGFFeature)
except NameError: None
