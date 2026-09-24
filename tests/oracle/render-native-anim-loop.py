# Native oracle: Scene.loopAnimations = False. animated_cubes' keys end at 11.25s, so t=12.5/14
# (frames 125/140 at 10 fps) sample each clip's own post-infinity behavior instead of the loop.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/VBufferRT.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/animated_cubes/animated_cubes.pyscene"))
m.scene.loopAnimations = False
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.framerate = 10
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-anim-loop"  # the test keeps only the depth captures

for frame in [125, 140]:
    m.clock.frame = frame
    m.renderFrame()
    m.frameCapture.capture()
exit()
