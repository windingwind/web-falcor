# Native oracle: runtime scene edits after load — light intensity + material
# baseColor/roughness changed via the python API, then one frame rendered
# (upstream MinimalPathTracer.py graph over the DoF oracle scene).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/MinimalPathTracer.py")).read())

m.loadScene(os.path.join(base, "assets/oracle-dof.pyscene"))

l = m.scene.getLight(0)
l.intensity = float3(30.0, 18.0, 6.0)
# getMaterial(name) overload is broken in this build (SystemError); pick from the list.
mat = [x for x in m.scene.materials if x.name == 'Mid'][0]
mat.baseColor = float4(0.9, 0.4, 0.1, 1.0)
mat.roughness = 0.1

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-runtime-edits"

m.renderFrame()
m.frameCapture.capture()
exit()
