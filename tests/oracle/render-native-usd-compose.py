# Native oracle: USD composition (assets/usd-compose.pyscene) through
# GBufferRT hit mask, world positions, normals and diffuse albedo.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW", "diffuseOpacity"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/usd-compose.pyscene"))
m.resizeFrameBuffer(192, 96)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-compose"
m.renderFrame()
m.frameCapture.capture()
exit()
