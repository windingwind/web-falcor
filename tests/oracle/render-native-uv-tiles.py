# Native oracle for Scene.getGeometryUVTiles: dumps each mesh's tiles (min/max points) as JSON.
# Run with the native falcor module (packman python) from tests/oracle/assets:
#   PYTHONPATH=<Release>/python LD_LIBRARY_PATH=<Release>:<packman python>/lib python3 render-native-uv-tiles.py <out.json>
import falcor, sys, json
out = sys.argv[1] if len(sys.argv) > 1 else "uv-tiles.json"
device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
testbed = falcor.Testbed(width=64, height=64, create_window=False, device=device)
testbed.load_scene("uv-tiles.pyscene")
testbed.frame()
result = []
for geometryID in range(2):
    tiles = testbed.scene.getGeometryUVTiles(geometryID)
    result.append([[t.min_point.x, t.min_point.y, t.max_point.x, t.max_point.y] for t in tiles])
with open(out, "w") as f:
    json.dump(result, f, indent=1)
