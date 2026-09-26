# Native oracle: one of Falcor/scripts/*.py (kScript) run unchanged on a production scene (kScene, a media path) at
# 320x180, the clock paused at t = 0, capturing the graph's marked outputs after kFrames frames into
# out-native/big-scripts/<kScript>/ (<kScript>-<scene name> for scenes other than BistroExterior).
# Per run, substitute kScript/kScene in a copy beside this file and run it with Mogwai --script <copy> --headless.
from falcor import *
import os

kScript = "RTXDI"
kScene = "Bistro_v5_2/BistroExterior.pyscene"
kFrames = 16

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
exec(open(os.path.join(root, "scripts", kScript + ".py")).read())
m.clock.pause()
m.loadScene(os.path.join(root, "media", kScene))
m.clock.time = 0
m.resizeFrameBuffer(320, 180)
m.ui = False
sceneName = os.path.splitext(os.path.basename(kScene))[0]
m.frameCapture.outputDir = os.path.join(base, "out-native", "big-scripts", kScript if sceneName == "BistroExterior" else kScript + "-" + sceneName)
m.frameCapture.baseFilename = kScript
for i in range(kFrames):
    m.renderFrame()
m.frameCapture.capture()
exit()
