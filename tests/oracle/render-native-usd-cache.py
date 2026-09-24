# Native oracle: time-sampled USD points (assets/usd-cache.pyscene) as vertex
# caches, captured through GBufferRT at 0.125s, 0.5s, 1.5s and 1.75s. The scene
# is reloaded per frame: with a paused clock, later captures of one load repeat the
# first frame's pose.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW", "texC"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.resizeFrameBuffer(256, 128)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-cache"
for frame in [3, 12, 36, 42]:
    m.loadScene(os.path.join(base, "assets/usd-cache.pyscene"))
    m.clock.framerate = 24
    m.clock.pause()
    m.clock.frame = frame
    m.renderFrame()
    m.frameCapture.capture()
exit()
