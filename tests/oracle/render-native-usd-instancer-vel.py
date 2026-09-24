# Native oracle: PointInstancers with velocities (assets/usd-instancer-vel.pyscene),
# captured through GBufferRT at 0.5s, 1.25s and 1.83s. The scene
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

m.resizeFrameBuffer(192, 96)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "usd-instancer-vel"
for frame in [12, 30, 44]:
    m.loadScene(os.path.join(base, "assets/usd-instancer-vel.pyscene"))
    m.clock.framerate = 24
    m.clock.pause()
    m.clock.frame = frame
    m.renderFrame()
    m.frameCapture.capture()
exit()
