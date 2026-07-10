# Native oracle: the upstream ToneMapping.py graph with physical exposure
# props (the test_ToneMapping.py fNumber/shutter variants).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/ToneMapping.py")).read())

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()
m.frameCapture.outputDir = os.path.join(base, "out-native")

for name, props in [("fnumber", {'autoExposure': False, 'fNumber': 0.5}),
                    ("shutter", {'autoExposure': False, 'fNumber': 1.0, 'shutter': 2.0})]:
    ToneMapping.updatePass('ToneMapping', props)
    m.frameCapture.baseFilename = "oracle-imgtest-tonemap-" + name
    m.renderFrame()
    m.frameCapture.capture()
exit()
