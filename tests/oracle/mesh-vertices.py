# Scene.get_mesh_vertices_and_indices / set_mesh_vertices on the Cornell box, run unchanged by native python
# (tests/oracle/out-native/mesh-vertices.json + .exr) and the web Testbed. The small box is moved and stretched;
# the image is captured before the edit (native keeps static BLASes, so its render after it hits the old box).
import falcor, json, sys
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "mesh_vertices"
device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False) if hasattr(falcor.DeviceType, "Vulkan") else falcor.Device()
testbed = falcor.Testbed(width=128, height=128, create_window=False, device=device)
g = testbed.create_render_graph("PathTracer")
g.create_pass("PathTracer", "PathTracer", {'samplesPerPixel': 1})
g.create_pass("VBufferRT", "VBufferRT", {'samplePattern': 'Stratified', 'sampleCount': 16})
g.create_pass("AccumulatePass", "AccumulatePass", {'enabled': True, 'precisionMode': 'Single'})
g.add_edge("VBufferRT.vbuffer", "PathTracer.vbuffer")
g.add_edge("PathTracer.color", "AccumulatePass.input")
g.mark_output("AccumulatePass.output")
testbed.render_graph = g
testbed.load_scene("test_scenes/cornell_box.pyscene")
scene = testbed.scene
for i in range(32): testbed.frame()
testbed.capture_output(out + ".before.exr")

flags = falcor.ResourceBindFlags.ShaderResource | falcor.ResourceBindFlags.UnorderedAccess
def buffers(vertex_count, triangle_count):
    return {k: device.create_structured_buffer(struct_size=12, element_count=triangle_count if k == "triangleIndices" else vertex_count, bind_flags=flags)
            for k in ["triangleIndices", "positions", "normals", "tangents", "texcrds"]}

def read(mesh_id):
    mesh = scene.get_mesh(mesh_id)
    b = buffers(mesh.vertex_count, mesh.triangle_count)
    scene.get_mesh_vertices_and_indices(mesh_id, b)
    f = lambda k, t: b[k].to_numpy().view(t).reshape(-1, 3)
    return f("triangleIndices", np.uint32), f("positions", np.float32), f("texcrds", np.float32), b

mesh_count = scene.stats["meshCount"]
before = [read(i)[:3] for i in range(mesh_count)]
# The small box: the last 24-vertex mesh. Flat normals and edge tangents per triangle.
box = max(i for i in range(mesh_count) if len(before[i][1]) == 24)
idx, pos, uv, b = read(box)
pos = pos * np.array([1.0, 1.5, 1.0], np.float32) + np.array([0.15, 0.0, 0.0], np.float32)
nrm = np.zeros_like(pos)
tan = np.zeros_like(pos)
for t in idx:
    p0, p1, p2 = pos[t[0]], pos[t[1]], pos[t[2]]
    n = np.cross(p1 - p0, p2 - p0)
    e = p1 - p0
    nrm[t] = n / np.linalg.norm(n)
    tan[t] = e / np.linalg.norm(e)
b["positions"].from_numpy(pos.astype(np.float32))
b["normals"].from_numpy(nrm.astype(np.float32))
b["tangents"].from_numpy(tan.astype(np.float32))
b["texcrds"].from_numpy(uv.astype(np.float32))
scene.set_mesh_vertices(box, b)
after = read(box)[:3]
for i in range(32): testbed.frame()
testbed.capture_output(out + ".after.exr")

result = {"meshCount": mesh_count, "box": box, "before": [[a.tolist() for a in m] for m in before], "after": [a.tolist() for a in after]}
if len(sys.argv) > 2:
    json.dump(result, open(sys.argv[2], "w"))
print("MESHVERTICES " + json.dumps(result))
