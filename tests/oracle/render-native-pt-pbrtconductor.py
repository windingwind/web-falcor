# Native oracle: VBufferRT -> MinimalPathTracer over oracle-pt-pbrtconductor.pyscene.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OraclePT")
vb = createPass("VBufferRT", {'useAlphaTest': False, 'samplePattern': 'Center'})
g.addPass(vb, "VBufferRT")
pt = createPass("MinimalPathTracer", {'maxBounces': 3})
g.addPass(pt, "MinimalPathTracer")
g.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer")
g.addEdge("VBufferRT.viewW", "MinimalPathTracer.viewW")
g.markOutput("MinimalPathTracer.color")
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/oracle-pt-pbrtconductor.pyscene"))
m.scene.camera.position = float3(0.5, 0.5, 2.0)
m.scene.camera.target = float3(0.5, 0.5, -1.0)
m.scene.camera.up = float3(0.0, 1.0, 0.0)

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-pt-pbrtconductor"

m.renderFrame()
m.frameCapture.capture()
exit()
