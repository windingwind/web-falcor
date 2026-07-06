# Native oracle: the UNMODIFIED upstream PathTracerDielectrics.py image-test
# graph over its own upstream test scene (nested_dielectrics.pyscene).
# 4 frames: stratified VBufferRT jitter (frames 2..4) + accumulation.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/PathTracerDielectrics.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/nested_dielectrics.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-dielectrics"

for i in range(4):
    m.renderFrame()
m.frameCapture.capture()
exit()
