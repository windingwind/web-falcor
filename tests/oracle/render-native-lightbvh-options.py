# Native oracle: full PathTracer with NON-DEFAULT LightBVH sampler options
# (Equal splits, 1 tri/leaf, no bounding cone, BoxToAverage bound) over the
# emissive two-quad scene. Same options on the web side -> bit-matched RNG.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OracleLightBVHOptions")
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
        'buildOptions': {'maxTriangleCountPerLeaf': 1, 'splitHeuristicSelection': 'Equal', 'useLeafCreationCost': False},
    },
})
g.addPass(pt, "PathTracer")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
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
m.frameCapture.baseFilename = "oracle-lightbvh-options"

m.renderFrame()
m.frameCapture.capture()
exit()
