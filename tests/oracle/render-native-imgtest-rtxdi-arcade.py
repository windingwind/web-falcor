# Native oracle: EXACT replica of the upstream image test
# tests/image_tests/renderpasses/test_RTXDI.py -- the unmodified RTXDI.py
# graph (VBufferRT -> RTXDIPass -> ToneMapper) over the upstream
# Arcade.pyscene, 640x360, captured at frames 1/16/64 (render_frames).
# RTXDIPass.color is additionally marked so the comparison runs on the
# linear pre-tonemap radiance (the web test marks it identically).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/RTXDI.py")).read())
RTXDI.markOutput("RTXDIPass.color")

m.loadScene(os.path.join(root, "media/Arcade/Arcade.pyscene"))

m.resizeFrameBuffer(640, 360)
m.ui = False
m.clock.framerate = 60
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-rtxdi-arcade"

frame = 0
for capture_frame in (1, 16, 64):
    while frame < capture_frame:
        frame += 1
        m.clock.frame = frame
        m.renderFrame()
    m.frameCapture.capture()
exit()
