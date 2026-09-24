# Native oracle: scripted animation (tests/oracle/assets/oracle-anim-script.pyscene). Records the
# animated light's and camera's exact positions per time, and captures GBufferRT posW at t=1.3 / 5.2.
from falcor import *
import os, json

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("AnimScript")
g.addPass(createPass("GBufferRT"), "GBufferRT")
g.markOutput("GBufferRT.posW")
m.addGraph(g)
m.loadScene(os.path.join(base, "assets/oracle-anim-script.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.framerate = 10
m.clock.pause()
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-anim-script"

times = [0.0, 0.5, 1.3, 2.7, 3.6, 5.2, 7.9]
record = []
for t in times:
    m.clock.frame = int(round(t * 10))
    m.renderFrame()
    l = m.scene.lights[0].position
    c = m.scene.camera.position
    ct = m.scene.camera.target
    record.append({"time": t, "light": [l.x, l.y, l.z], "camera": [c.x, c.y, c.z], "target": [ct.x, ct.y, ct.z]})
    if m.clock.frame in (13, 52):
        m.frameCapture.capture()
with open(os.path.join(base, "out-native/oracle-anim-script.json"), "w") as f:
    json.dump(record, f, indent=1)
exit()
