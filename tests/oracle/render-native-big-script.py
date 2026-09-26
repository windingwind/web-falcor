# Native oracle: one of Falcor/scripts/*.py (kScript) run unchanged on BistroExterior at 320x180, the clock paused at
# t = 0, capturing the graph's marked outputs after kFrames frames into out-native/big-scripts/<kScript>/.
# Per graph, substitute kScript in a copy beside this file and run it with Mogwai --script <copy> --headless.
from falcor import *
import os

kScript = "RTXDI"
kFrames = 16

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
exec(open(os.path.join(root, "scripts", kScript + ".py")).read())
m.clock.pause()
m.loadScene(os.path.join(root, "media/Bistro_v5_2/BistroExterior.pyscene"))
m.clock.time = 0
m.resizeFrameBuffer(320, 180)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native", "big-scripts", kScript)
m.frameCapture.baseFilename = kScript
for i in range(kFrames):
    m.renderFrame()
m.frameCapture.capture()
exit()
