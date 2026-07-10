# Native oracle: object motion vectors from the upstream VBufferRT.py graph
# over animated scenes — cesium_man (skinned, prevVertices path; native writes
# ZERO skinned mvecs on this content, so only geometry is oracled) and
# animated_cubes (rigid, prevWorldMatrices path; sampled INSIDE the key range
# 6.25..11.25s so no AnimationBehavior extrapolation is involved).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())

for scene, frames in [("cesium_man/CesiumMan.pyscene", (0, 1)), ("animated_cubes/animated_cubes.pyscene", (63, 64))]:
    m.loadScene(os.path.join(root, "media/test_scenes", scene))
    m.resizeFrameBuffer(256, 256)
    m.ui = False
    m.clock.framerate = 10
    m.clock.time = 0
    m.clock.pause()

    m.frameCapture.outputDir = os.path.join(base, "out-native")
    m.frameCapture.baseFilename = "oracle-mvec-" + scene.split("/")[0]

    m.clock.frame = frames[0]
    m.renderFrame()
    m.frameCapture.capture()
    m.clock.frame = frames[1]
    m.renderFrame()
    m.frameCapture.capture()
exit()
