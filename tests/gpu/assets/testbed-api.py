# Testbed python API (Falcor/Source/Falcor/Core/Testbed.cpp bindings): load_scene_from_string,
# get_import_paths/dicts, load_render_graph, window, and the keyboard/mouse/resize callbacks.
import falcor

testbed = falcor.Testbed(width=64, height=64, create_window=False)
testbed.load_scene_from_string("""
m = StandardMaterial('M')
sceneBuilder.addMeshInstance(sceneBuilder.addNode('q', Transform()), sceneBuilder.addTriangleMesh(TriangleMesh.createQuad(), m))
""")
assert testbed.get_import_paths() == ['<memory>'], testbed.get_import_paths()
assert testbed.get_import_dicts() == [{}]
assert testbed.window is None

g = testbed.load_render_graph('/Falcor/tests/image_tests/renderpasses/graphs/ToneMapping.py')
testbed.render_graph = g
assert g['ToneMapping'] is not None and g.name == 'ToneMapper'

events = []
def on_key(e):
    events.append((e.type, e.key))
    if e.type == falcor.KeyboardEvent.Type.KeyPressed and e.key == falcor.Key.E:
        testbed.show_ui = False
    return True
testbed.keyboard_event_callback = on_key
sizes = []
testbed.window_size_change_callback = lambda w, h: sizes.append((w, h))
testbed.resize_frame_buffer(32, 16)
assert sizes == [(32, 16)], sizes
testbed.frame()

# Device binding.
device = testbed.device
assert device.info.api_name == "WebGPU" and isinstance(device.info.adapter_name, str)
assert device.limits.max_compute_dispatch_thread_groups.x >= 65535
assert len(falcor.Device.get_gpus()) == 1
sampler = device.create_sampler(mag_filter=falcor.TextureFilteringMode.Point, address_mode_u=falcor.TextureAddressingMode.Clamp)
assert sampler is not None
device.end_frame()
device.wait()

# Buffer and Texture properties.
sb = device.create_structured_buffer(struct_size=16, element_count=4)
assert sb.is_structured and not sb.is_typed and sb.memory_type == falcor.MemoryType.DeviceLocal
tb = device.create_typed_buffer(falcor.ResourceFormat.R32Float, 8)
assert tb.is_typed and tb.format == falcor.ResourceFormat.R32Float
tex = device.create_texture(8, 4, format=falcor.ResourceFormat.RGBA8Unorm, array_size=2, mip_levels=1)
assert tex.format == falcor.ResourceFormat.RGBA8Unorm and tex.array_size == 2 and tex.sample_count == 1
