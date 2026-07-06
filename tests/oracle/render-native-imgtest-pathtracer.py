# Native oracle: the UNMODIFIED upstream PathTracer.py image-test graph over
# cornell_box.pyscene (upstream test uses Arcade.pyscene -> FBX importer
# pending on web; the graph itself is untouched). 4 accumulated frames.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/PathTracer.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-pathtracer"

for i in range(4):
    m.renderFrame()
m.frameCapture.capture()
exit()
