# Native oracle: USD import — oracle-usd.usda (meshes + UsdPreviewSurface
# materials + SphereLight) rendered through the upstream MinimalPathTracer.py
# graph. Requires the USDImporter plugin (build/.../plugins/USDImporter.so —
# re-enabled from plugins-disabled/ on this machine).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).read())

m.loadScene(os.path.join(base, "assets/oracle-usd.usda"))

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-usd"

m.renderFrame()
m.frameCapture.capture()
exit()
