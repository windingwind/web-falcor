# Native oracle: upstream SceneDebugger.py graph over the smoke volume scene.
# The scene mirrors Falcor/media/test_scenes/smoke.pyscene exactly, except the
# density grid loads from the pre-converted smoke.nvdb (the prebuilt native
# openvdb segfaults in initialize() on this machine, so native cannot read
# .vdb; the NanoVDB buffer is byte-identical to what the web builds from the
# original smoke.vdb - tools/vdb/, tests/oracle/validate-nvdb.py). The scene
# is generated here with absolute paths (native asset resolution rejects
# ../-relative paths).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

scene = os.path.join(base, "assets", "smoke-nvdb-generated.pyscene")
open(scene, "w").write(f"""
smokeVolume = GridVolume('smoke')
smokeVolume.loadGrid(GridVolume.GridSlot.Density, '{os.path.join(base, "assets", "smoke.nvdb")}', 'density')
smokeVolume.densityScale = 0.5
smokeVolume.albedo = float3(0.5, 0.5, 0.5)
sceneBuilder.addGridVolume(smokeVolume)

camera = Camera()
camera.position = float3(25, 0, 55)
camera.target = float3(0, 25, 0)
camera.up = float3(0, 1, 0)
sceneBuilder.addCamera(camera)

sceneBuilder.envMap = EnvMap('{os.path.join(root, "media", "test_scenes", "envmaps", "20060807_wells6_hd.hdr")}')
sceneBuilder.envMap.intensity = 1.5
""")

exec(open(os.path.join(root, "tests/image_tests/renderpasses/graphs/SceneDebugger.py")).read())

m.loadScene(scene)
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-feature-smoke-debugger"

m.renderFrame()
m.frameCapture.capture()
exit()
