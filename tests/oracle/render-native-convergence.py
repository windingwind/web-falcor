# Native oracle: PathTracer over the UNMODIFIED upstream convergence_test.pyscene.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
media = os.path.abspath(os.path.join(base, "../../Falcor/media/test_scenes"))

g = RenderGraph("OracleSphereArray")
vb = createPass("VBufferRT", {'useAlphaTest': False, 'samplePattern': 'Center'})
g.addPass(vb, "VBufferRT")
pt = createPass("PathTracer", {
    'samplesPerPixel': 1,
    'maxSurfaceBounces': 3, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 10,
    'useRussianRoulette': False,
    'emissiveSampler': 'LightBVH',
    'useSER': False,
})
g.addPass(pt, "PathTracer")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.markOutput("PathTracer.color")
m.addGraph(g)

m.loadScene(os.path.join(media, "convergence_test.pyscene"))

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-convergence"

m.renderFrame()
m.frameCapture.capture()
exit()
