# Native oracle: per-clip pre-infinity AnimationBehavior extrapolation.
# animated_cubes.pyscene assigns Cycle/Cycle/Linear/Oscillate to clips 1..4
# (green cube 0 stays Constant); its keys span 6.25..11.25s, so t=3.0/5.5
# (frames 30/55 at 10 fps) sample the PRE-infinity region where each cube's
# pose depends on its behavior (Constant parks at the first key).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/animated_cubes/animated_cubes.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.framerate = 10
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-anim-behaviors"

m.clock.frame = 30
m.renderFrame()
m.frameCapture.capture()
m.clock.frame = 55
m.renderFrame()
m.frameCapture.capture()
exit()
