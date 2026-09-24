# Native oracle for Scene.stats: loads test scenes in a Testbed and dumps each scene's stats dict as JSON.
# Run with the native falcor module (CPython 3.10, packman) from Falcor/media:
#   PYTHONPATH=<Release>/python LD_LIBRARY_PATH=<Release>:<packman python>/lib python3 render-native-scene-stats.py <out.json>
import falcor, sys, json
out = sys.argv[1] if len(sys.argv) > 1 else "scene-stats.json"
scenes = ["test_scenes/cornell_box.pyscene", "test_scenes/geometry_types.pyscene", "test_scenes/material_test.pyscene"]
device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
testbed = falcor.Testbed(width=64, height=64, create_window=False, device=device)
result = {}
for path in scenes:
    testbed.load_scene(path)
    testbed.frame()
    result[path] = dict(testbed.scene.stats)
with open(out, "w") as f:
    json.dump(result, f, indent=1, sort_keys=True)
