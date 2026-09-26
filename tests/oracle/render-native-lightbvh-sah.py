# Native oracle: full PathTracer with a BinnedSAH LightBVH (2 tris/leaf, leaf creation cost)
# over three emissive spheres. The web runs this script unchanged (MogwaiScriptRunner).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OracleLightBVHSAH")
vb = createPass("VBufferRT", {'useAlphaTest': False, 'samplePattern': 'Center'})
g.addPass(vb, "VBufferRT")
pt = createPass("PathTracer", {
    'samplesPerPixel': 1,
    'maxSurfaceBounces': 3, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 10,
    'useRussianRoulette': False,
    'emissiveSampler': 'LightBVH',
    'useSER': False,
    'lightBVHOptions': {
        'useBoundingCone': False,
        'solidAngleBoundMethod': 'BoxToAverage',
        'buildOptions': {'maxTriangleCountPerLeaf': 2, 'splitHeuristicSelection': 'BinnedSAH', 'useLeafCreationCost': True},
    },
})
g.addPass(pt, "PathTracer")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.markOutput("PathTracer.color")
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/oracle-lightbvh-sah.pyscene"))
m.scene.camera.position = float3(0.5, 0.6, 1.6)
m.scene.camera.target = float3(0.5, 0.0, -0.7)
m.scene.camera.up = float3(0.0, 1.0, 0.0)

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-lightbvh-sah"

m.renderFrame()
m.frameCapture.capture()
exit()
