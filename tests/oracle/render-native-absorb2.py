# Native oracle for the volume-absorption isolation scene.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("Absorb")
g.addPass(createPass("VBufferRT", {'samplePattern': 'Center'}), "VBufferRT")
g.addPass(createPass("PathTracer", {'samplesPerPixel': 1}), "PathTracer")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.markOutput("PathTracer.color")
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/oracle-absorb2.pyscene"))
m.resizeFrameBuffer(128, 128)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-absorb2"

m.renderFrame()
m.frameCapture.capture()
exit()
