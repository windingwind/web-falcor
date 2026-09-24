# Native oracle: PathTracer's NRD outputs (the guide buffers NRDPass consumes) over Arcade, the
# scene of the upstream test_PathTracerNRD.py, with PathTracerNRD.py's path tracer settings.
# GBufferRT samples pixel centers; the five radiance outputs accumulate over 64 frames, the
# other outputs are captured as rendered (they don't depend on the sample sequence much). All
# captures keep alpha (hit distances, material IDs).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
radiance = ["nrdDiffuseRadianceHitDist", "nrdSpecularRadianceHitDist", "nrdDeltaReflectionRadianceHitDist", "nrdDeltaTransmissionRadianceHitDist", "nrdResidualRadianceHitDist"]
guides = ["nrdEmission", "nrdDiffuseReflectance", "nrdSpecularReflectance", "nrdDeltaReflectionReflectance", "nrdDeltaReflectionEmission",
          "nrdDeltaReflectionNormWRoughMaterialID", "nrdDeltaReflectionPathLength", "nrdDeltaReflectionHitDist", "nrdDeltaTransmissionReflectance",
          "nrdDeltaTransmissionEmission", "nrdDeltaTransmissionNormWRoughMaterialID", "nrdDeltaTransmissionPathLength", "nrdDeltaTransmissionPosW"]

g = RenderGraph("PathTracerNRDOutputs")
g.addPass(createPass("GBufferRT", {"samplePattern": "Center", "useAlphaTest": True}), "GBufferRT")
g.addPass(createPass("PathTracer", {"samplesPerPixel": 1, "maxSurfaceBounces": 10, "useRussianRoulette": True}), "PathTracer")
g.addEdge("GBufferRT.vbuffer", "PathTracer.vbuffer")
g.addEdge("GBufferRT.viewW", "PathTracer.viewW")
for name in radiance:
    g.addPass(createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"}), "Acc_" + name)
    g.addEdge("PathTracer." + name, "Acc_" + name + ".input")
    g.markOutput("Acc_" + name + ".output", TextureChannelFlags.RGBA)
for name in guides:
    if name.endswith("NormWRoughMaterialID"):
        # RGB10A2Unorm captures don't round-trip: blit to RGBA32Float first.
        g.addPass(createPass("BlitPass", {"outputFormat": "RGBA32Float"}), "Blit_" + name)
        g.addEdge("PathTracer." + name, "Blit_" + name + ".src")
        g.markOutput("Blit_" + name + ".dst", TextureChannelFlags.RGBA)
    else:
        g.markOutput("PathTracer." + name, TextureChannelFlags.RGBA)
# PathTracer::beginFrame dereferences the color output: it must be allocated.
g.markOutput("PathTracer.color")
m.addGraph(g)
m.loadScene(os.path.join(root, "media/Arcade/Arcade.pyscene"))
m.resizeFrameBuffer(320, 180)
m.ui = False
m.clock.framerate = 60
m.clock.time = 0
m.clock.pause()
m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "ptnrd"
for frame in range(1, 65):
    m.clock.frame = frame
    m.renderFrame()
m.frameCapture.capture()
exit()
