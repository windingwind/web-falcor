# Native oracle: the UNMODIFIED upstream image-test graph
# (tests/image_tests/renderpasses/graphs/VBufferRT.py) over the
# unmodified upstream cornell_box.pyscene.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box.pyscene"))

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-vbuffer"

m.renderFrame()
m.frameCapture.capture()
exit()
