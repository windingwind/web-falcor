# Native oracle: upstream VBufferRT.py graph over the thin-lens DoF scene
# (camera apertureRadius 0.05, focalDistance 3.05).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())

m.loadScene(os.path.join(base, "assets/oracle-dof.pyscene"))

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-dof"

m.renderFrame()
m.frameCapture.capture()
exit()
