# Native oracle: runtime emissive edit — cornell_box light dimmed to
# emissiveFactor 0.4 after load, PathTracer 64 accumulated frames.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/PathTracer.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box.pyscene"))

mat = [x for x in m.scene.materials if x.name == 'Light'][0]
mat.emissiveFactor = 0.4

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-emissive-edit"
for i in range(64):
    m.renderFrame()
m.frameCapture.capture()
exit()
