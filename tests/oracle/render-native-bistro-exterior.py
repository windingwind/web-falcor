# Native oracle: BistroExterior.pyscene (the media drop's 120 MB FBX, 405 mostly BC
# textures) through GBufferRT: geometry and the full-resolution BC material textures.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW", "texC", "diffuseOpacity", "specRough"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.loadScene(os.path.join(root, "media/Bistro_v5_2/BistroExterior.pyscene"))
m.resizeFrameBuffer(320, 180)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "bistro-exterior"
m.renderFrame()
m.frameCapture.capture()
exit()
