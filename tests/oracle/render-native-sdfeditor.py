# Native oracle: the upstream SDFEditor image test (graphs/SDFEditorRenderGraphV2.py over
# assets/sdfeditor-two-grids.pyscene, the upstream scene with cheese grids; 64 frames at 640x360).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/SDFEditorRenderGraphV2.py")).read())
m.loadScene(os.path.join(base, "assets/sdfeditor-two-grids.pyscene"))
# The editor's input too: the GUI layer is compared over it (native's SBS surface is unusable here).
DefaultRenderGraph.markOutput("ToneMapper.dst")
m.resizeFrameBuffer(640, 360)
m.ui = False
m.clock.framerate = 60
m.clock.time = 0
m.clock.pause()
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "sdfeditor"
for frame in range(1, 65):
    m.clock.frame = frame
    m.renderFrame()
m.frameCapture.capture()
exit()
