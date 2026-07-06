# TAA feature graph (web-falcor): mirrors the upstream TAA.py image test 1:1
# except GBufferRaster -> GBufferRT (this machine's native driver lacks ROV,
# so GBufferRaster cannot produce a native oracle; GBufferRT channels are
# verified elsewhere). Halton camera jitter drives per-frame varying input
# through the UNMODIFIED TAA kernel: YCgCo neighborhood clamp, Catmull-Rom
# history sampling and anti-flicker are all exercised.
from falcor import *

def render_graph_TAAFeature():
    g = RenderGraph("TAAFeature")
    GBufferRT = createPass("GBufferRT", {"samplePattern": 'Halton'})
    TAAPass = createPass("TAA")
    g.addPass(GBufferRT, "GBufferRT")
    g.addPass(TAAPass, "TAA")
    g.addEdge("GBufferRT.diffuseOpacity", "TAA.colorIn")
    g.addEdge("GBufferRT.mvec", "TAA.motionVecs")
    g.markOutput("TAA.colorOut")
    return g

TAAFeature = render_graph_TAAFeature()
try: m.addGraph(TAAFeature)
except NameError: None
