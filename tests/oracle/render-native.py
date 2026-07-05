# Native oracle render: GBufferRaster over the shared quad.gltf.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

g = RenderGraph("OracleGBuffer")
# Native GBufferRaster needs ROVs (unavailable on Linux/Vulkan); GBufferRT
# produces identical posW/texC for opaque geometry.
gb = createPass("GBufferRT")
g.addPass(gb, "GBufferRT")
g.markOutput("GBufferRT.texC")
g.markOutput("GBufferRT.posW")
m.addGraph(g)

m.loadScene(os.path.join(base, "assets/quad.gltf"))
m.scene.camera.position = float3(0.5, 0.5, 2.0)
m.scene.camera.target = float3(0.5, 0.5, -1.0)
m.scene.camera.up = float3(0.0, 1.0, 0.0)

m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle"

m.renderFrame()
m.frameCapture.capture()
exit()
