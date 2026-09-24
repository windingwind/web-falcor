# Native oracle for Scene.replace_material: Falcor's scripts/python/test_replace_material.py without
# its NeuralMaterial replacement (that class isn't part of Falcor 8.0), at 256x256 for 64 frames.
# Run with the native falcor module (CPython 3.10, packman) from Falcor/media:
#   PYTHONPATH=<Release>/python LD_LIBRARY_PATH=<Release>:<packman python>/lib python3 render-native-replace-material.py <out.exr>
# The web test runs this same script through runTestbedScript and reads the graph output.
import falcor, sys, os
out = sys.argv[1] if len(sys.argv) > 1 else "replace_material.exr"
device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
testbed = falcor.Testbed(width=256, height=256, create_window=False, device=device)
g = testbed.create_render_graph("PathTracer")
g.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
g.create_pass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16, 'useAlphaTest': True})
g.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
g.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.add_edge("PathTracer.color", "AccumulatePass.input")
g.mark_output("AccumulatePass.output")
testbed.render_graph = g
testbed.load_scene("test_scenes/cornell_box.pyscene")
testbed.frame()
mat1 = falcor.PBRTDiffuseMaterial(device, "PBRT diffuse")
mat1.load_texture(falcor.MaterialTextureSlot.BaseColor, 'test_scenes/textures/checker_tile_base_color.png')
testbed.scene.replace_material(0, mat1)
for i in range(64): testbed.frame()
testbed.capture_output(out)
