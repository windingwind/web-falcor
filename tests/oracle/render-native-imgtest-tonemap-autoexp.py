# Native oracle: the upstream ToneMapping.py image-test graph with
# autoExposure enabled (the test_ToneMapping.py 'autoExposure.True' variant).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/ToneMapping.py")).read())

ToneMapping.updatePass('ToneMapping', {'autoExposure': True})

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-tonemap-autoexp"

m.renderFrame()
m.frameCapture.capture()
exit()
