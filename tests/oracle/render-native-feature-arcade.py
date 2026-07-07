# Native oracle: upstream GBufferRT.py image-test graph over the upstream
# Arcade.pyscene (FBX import) - validates the web FBX importer end-to-end
# (geometry transforms, material mapping, textures) via GBuffer channels.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/GBufferRT.py")).read())

m.loadScene(os.path.join(root, "media/Arcade/Arcade.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-feature-arcade"

m.renderFrame()
m.frameCapture.capture()
exit()
