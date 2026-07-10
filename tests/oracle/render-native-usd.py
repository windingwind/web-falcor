# Native oracle: USD import — oracle-usd.usda (meshes + UsdPreviewSurface
# materials + SphereLight + camera) rendered through the upstream
# MinimalPathTracer.py graph (radiance) and VBufferRT.py (deterministic
# depth/mask for geometry/camera parity). Requires the USDImporter plugin
# (build/.../plugins/USDImporter.so — re-enabled from plugins-disabled/).
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

# 64 accumulated frames: NEE sample sequences decorrelate across
# implementations, the average converges.
for i in range(64):
    m.renderFrame()
m.frameCapture.capture()

m.removeGraph(MinimalPathTracer)
exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())
m.resizeFrameBuffer(256, 256)
m.frameCapture.baseFilename = "oracle-usd-vbuffer"
m.renderFrame()
m.frameCapture.capture()
exit()
