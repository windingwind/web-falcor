# Scene.get_material_params / set_material_params on the Cornell box, run unchanged by native python
# (tests/oracle/out-native/material-params.json) and the web Testbed.
import falcor, json, sys
import numpy as np

device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False) if hasattr(falcor.DeviceType, "Vulkan") else falcor.Device()
testbed = falcor.Testbed(width=64, height=64, create_window=False, device=device)
testbed.load_scene("test_scenes/cornell_box.pyscene")
scene = testbed.scene
n = len(scene.materials)
ids = device.create_structured_buffer(struct_size=4, element_count=n, bind_flags=falcor.ResourceBindFlags.ShaderResource)
ids.from_numpy(np.arange(n, dtype=np.uint32))
params = device.create_buffer(size=n * 20 * 4, bind_flags=falcor.ResourceBindFlags.ShaderResource)
scene.get_material_params(ids, params)
before = params.to_numpy().view(np.float32).reshape(n, 20)
edited = before.copy()
edited[:, :3] = [0.2, 0.4, 0.6]  # base color
edited[:, 4:8] = 2.0  # out of range: clamped by deserializeParams
params.from_numpy(edited.astype(np.float32))
scene.set_material_params(ids, params)
scene.get_material_params(ids, params)
after = params.to_numpy().view(np.float32).reshape(n, 20)
result = {"names": [m.name for m in scene.materials], "before": before.tolist(), "after": after.tolist(),
          "count": falcor.IMaterial.PARAM_COUNT, "layouts": {k: dict(v) for k, v in falcor.MATERIAL_PARAM_LAYOUTS.items()},
          "standard": falcor.get_material_param_layout(falcor.MaterialType.Standard)}
out = sys.argv[1] if len(sys.argv) > 1 else None
if out:
    json.dump(result, open(out, "w"), indent=1)
print("MATERIALPARAMS " + json.dumps(result))
