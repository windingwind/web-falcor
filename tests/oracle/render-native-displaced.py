# Native oracle: displacement mapping — cornell_box_displaced.pyscene through
# SceneDebugger FaceNormal (deterministic) and MinimalPathTracer (64 frames).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

g = RenderGraph("SceneDebugger")
SceneDebugger = createPass("SceneDebugger", {'mode': 'FaceNormal'})
g.addPass(SceneDebugger, "SceneDebugger")
g.markOutput("SceneDebugger.output")
m.addGraph(g)

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box_displaced.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-displaced-debug"
m.renderFrame()
m.frameCapture.capture()

m.removeGraph(g)
exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).read())
m.resizeFrameBuffer(256, 256)
m.frameCapture.baseFilename = "oracle-displaced-mpt"
for i in range(64):
    m.renderFrame()
m.frameCapture.capture()
exit()
