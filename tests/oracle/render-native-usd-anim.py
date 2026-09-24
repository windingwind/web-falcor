# Native oracle: time-sampled USD xformOps (assets/usd-anim.pyscene) as node
# animations, captured through GBufferRT at 0.5s, 1.25s and 2.5s (looped). The scene
# is reloaded per frame: with a paused clock, later captures of one load repeat the
# first frame's pose.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)

m.resizeFrameBuffer(256, 128)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-anim"
for frame in [12, 30, 60]:
    m.loadScene(os.path.join(base, "assets/usd-anim.pyscene"))
    m.clock.framerate = 24
    m.clock.pause()
    m.clock.frame = frame
    m.renderFrame()
    m.frameCapture.capture()
exit()
