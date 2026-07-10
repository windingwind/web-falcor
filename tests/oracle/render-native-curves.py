# Native oracle: curve geometry (linear swept spheres) — the upstream
# two_curves.pyscene rendered through SceneDebugger GeometryID (deterministic)
# and MinimalPathTracer (shaded, 64 accumulated frames).
# Requires the USDImporter plugin (re-enabled from plugins-disabled/).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

g = RenderGraph("SceneDebugger")
SceneDebugger = createPass("SceneDebugger", {'mode': 'FaceNormal'})
g.addPass(SceneDebugger, "SceneDebugger")
g.markOutput("SceneDebugger.output")
m.addGraph(g)

m.loadScene(os.path.join(root, "media/test_scenes/curves/two_curves.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-curves-debug"
m.renderFrame()
m.frameCapture.capture()
exit()
