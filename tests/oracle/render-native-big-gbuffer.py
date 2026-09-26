# Native oracle: the production scenes besides BistroExterior (render-native-bistro-exterior.py) through GBufferRT
# from each scene's own camera at t = 0, into out-native/big-gbuffer/<scene>.GBufferRT.<channel>.0.exr.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))
scenes = {
    "BistroInterior_Wine": "Bistro_v5_2/BistroInterior_Wine.pyscene",
    "EmeraldSquare_Day": "EmeraldSquare_v4_1/EmeraldSquare_Day.pyscene",
    "SunTemple": "SunTemple_v4/SunTemple/SunTemple.pyscene",
    "ZeroDay": "ZeroDay_v1/ZeroDay.pyscene",
}

g = RenderGraph("GBufferRT")
g.addPass(createPass("GBufferRT", {"useTraceRayInline": True, "samplePattern": "Center"}), "GBufferRT")
for c in ["mask", "posW", "normW", "texC", "diffuseOpacity", "specRough", "emissive"]:
    g.markOutput("GBufferRT." + c)
m.addGraph(g)
m.resizeFrameBuffer(320, 180)
m.ui = False
m.frameCapture.outputDir = os.path.join(base, "out-native", "big-gbuffer")

for name, path in scenes.items():
    m.loadScene(os.path.join(root, "media", path))
    m.clock.time = 0
    m.clock.pause()
    m.frameCapture.baseFilename = name
    m.renderFrame()
    m.frameCapture.capture()
exit()
