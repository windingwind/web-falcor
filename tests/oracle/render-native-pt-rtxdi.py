# Native oracle: full PathTracer with USE_RTXDI (in-tracer ReSTIR direct
# lighting) over the emissive two-quad scene. 16 frames advance the temporal
# reservoir chain (RNG bit-matched on the web side), then the 16th frame's
# color is captured un-accumulated.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OraclePTRTXDI")
vb = createPass("VBufferRT", {'useAlphaTest': False, 'samplePattern': 'Center'})
g.addPass(vb, "VBufferRT")
pt = createPass("PathTracer", {
    'samplesPerPixel': 1,
    'maxSurfaceBounces': 3, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 10,
    'useRussianRoulette': False,
    'useSER': False,
    'useRTXDI': True,
})
g.addPass(pt, "PathTracer")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.addEdge("VBufferRT.mvec", "PathTracer.mvec")
g.markOutput("PathTracer.color")
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/oracle-pt-emissive.pyscene"))
m.scene.camera.position = float3(0.5, 0.5, 2.0)
m.scene.camera.target = float3(0.5, 0.5, -1.0)
m.scene.camera.up = float3(0.0, 1.0, 0.0)

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-pt-rtxdi"

for _ in range(16):
    m.renderFrame()
m.frameCapture.capture()
exit()
