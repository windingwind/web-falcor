# Native oracle: the production scenes through the default PathTracer (1 spp), accumulated over 64 frames at t = 0,
# into out-native/big-pt/<scene>.AccumulatePass.output.0.exr. Run with --headless.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
scenes = {
    "BistroExterior": "Bistro_v5_2/BistroExterior.pyscene",
    "BistroInterior_Wine": "Bistro_v5_2/BistroInterior_Wine.pyscene",
    "EmeraldSquare_Day": "EmeraldSquare_v4_1/EmeraldSquare_Day.pyscene",
    "SunTemple": "SunTemple_v4/SunTemple/SunTemple.pyscene",
    "ZeroDay": "ZeroDay_v1/ZeroDay.pyscene",
}
kFrames = 64

g = RenderGraph("BigPT")
g.addPass(createPass("VBufferRT", {"samplePattern": "Center", "useAlphaTest": True}), "VBufferRT")
g.addPass(createPass("PathTracer", {"samplesPerPixel": 1, "useSER": False}), "PathTracer")
g.addPass(createPass("AccumulatePass", {"enabled": True, "precisionMode": "Single"}), "AccumulatePass")
g.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.addEdge("PathTracer.color", "AccumulatePass.input")
g.markOutput("AccumulatePass.output")
m.addGraph(g)
m.resizeFrameBuffer(320, 180)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native", "big-pt")

for name, path in scenes.items():
    m.clock.pause()
    m.loadScene(os.path.join(root, "media", path))
    m.clock.time = 0
    m.frameCapture.baseFilename = name
    for frame in range(kFrames):
        m.renderFrame()  # the paused clock keeps t = 0; PathTracer advances its own frame count
    m.frameCapture.capture()
exit()
