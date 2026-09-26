# Native oracle: full PathTracer with the Uniform emissive sampler over three emissive spheres
# (their top-pole triangles are degenerate: zero flux, culled from the active list).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OracleEmissiveUniform")
vb = createPass("VBufferRT", {'useAlphaTest': False, 'samplePattern': 'Center'})
g.addPass(vb, "VBufferRT")
pt = createPass("PathTracer", {
    'samplesPerPixel': 1,
    'maxSurfaceBounces': 3, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 10,
    'useRussianRoulette': False,
    'emissiveSampler': 'Uniform',
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
m.frameCapture.baseFilename = "oracle-emissive-uniform"

m.renderFrame()
m.frameCapture.capture()
exit()
