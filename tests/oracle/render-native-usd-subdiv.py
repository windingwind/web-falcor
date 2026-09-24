# Native oracle: subdivision surfaces (assets/usd-subdiv.pyscene) through
# GBufferRT hit mask, positions, normals and texcoords.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW", "faceNormalW", "texC"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/usd-subdiv.pyscene"))
m.resizeFrameBuffer(256, 128)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-subdiv"
m.renderFrame()
m.frameCapture.capture()
exit()
