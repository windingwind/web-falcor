# Native oracle: WARDiffPathTracer in ForwardDiffDebug mode (upstream graphs
# WARDiffPathTracerTranslationFwd.py and WARDiffPathTracerMaterialFwd.py over
# bunny_war_diff_pt.pyscene with the upstream test's builder flags), 128x128, 64
# accumulated frames. Captures the accumulated primal and gradient images.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
graphs = os.path.join(root, "tests/image_tests/renderpasses/graphs")

flags = SceneBuilderFlags.DontMergeMaterials | SceneBuilderFlags.RTDontMergeDynamic | SceneBuilderFlags.DontOptimizeMaterials
m.resizeFrameBuffer(128, 128)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native")

for name in ["WARDiffPathTracerTranslationFwd", "WARDiffPathTracerMaterialFwd"]:
    m.removeAllGraphs() if hasattr(m, "removeAllGraphs") else None
    m.script(os.path.join(graphs, name + ".py"))
    m.loadScene(os.path.join(root, "media/test_scenes/bunny_war_diff_pt.pyscene"), buildFlags=flags)
    m.clock.framerate = 60
    m.clock.time = 0
    m.clock.pause()
    m.frameCapture.baseFilename = "wardiff-" + name
    for frame in range(1, 65):
        m.clock.frame = frame
        m.renderFrame()
    m.frameCapture.capture()
    m.removeGraph(m.activeGraph)
exit()
