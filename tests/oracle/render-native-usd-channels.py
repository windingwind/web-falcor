# Native oracle: UsdUVTexture channel selectors, sourceColorSpace, UsdTransform2d
# and value scale (assets/usd-channels.pyscene) through GBufferRT's texture and
# material channels. Requires the USDImporter plugin.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "texC", "diffuseOpacity", "specRough", "emissive"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/usd-channels.pyscene"))
m.resizeFrameBuffer(128, 64)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-channels"
m.renderFrame()
m.frameCapture.capture()
exit()
