# Native oracle: TAA feature graph (see graphs/taa-feature.py) over
# cornell_box.pyscene, 8 frames with Halton camera jitter (static camera —
# scripted per-frame camera moves are NOT reproducible in native Mogwai:
# the camera controller composes them statefully). Captures frame 8.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(base, "graphs/taa-feature.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-feature-taa"

for i in range(8):
    m.renderFrame()
m.frameCapture.capture()
exit()
