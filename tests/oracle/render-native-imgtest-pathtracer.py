# Native oracle: the UNMODIFIED upstream PathTracer.py image-test graph over
# cornell_box.pyscene (upstream test uses Arcade.pyscene -> FBX importer
# pending on web; the graph itself is untouched). 4 accumulated frames.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/PathTracer.py")).read())

m.loadScene(os.path.join(root, "media/test_scenes/cornell_box.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-imgtest-pathtracer"

for i in range(4):
    m.renderFrame()
m.frameCapture.capture()

# rayCount/pathLength are R32Uint; the native EXR capture path writes zeros
# for uint formats (Bitmap uint->EXR bug), so dump raw texture data instead.
# Requires numpy in the bundled python (pythondist/python3 -m pip install numpy).
import numpy as np
for name in ["rayCount", "pathLength"]:
    tex = m.activeGraph.getOutput("PathTracer." + name).to_numpy()
    tex.astype(np.uint32).tofile(os.path.join(base, "out-native", "oracle-imgtest-pathtracer.PathTracer." + name + ".0.u32"))
exit()
