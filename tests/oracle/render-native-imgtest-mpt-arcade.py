# Native oracle: EXACT replica of the upstream image test
# test_MinimalPathTracer.py -- unmodified MinimalPathTracer.py graph over the
# upstream Arcade.pyscene, 640x360, captured at frame 128.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).read())

m.loadScene(os.path.join(root, "media/Arcade/Arcade.pyscene"))
m.resizeFrameBuffer(640, 360)
m.ui = False
m.clock.framerate = 60
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-mpt-arcade"

frame = 0
while frame < 128:
    frame += 1
    m.clock.frame = frame
    m.renderFrame()
m.frameCapture.capture()
exit()
