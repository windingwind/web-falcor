# Native oracle for Scene.stats: loads test scenes in a Testbed and dumps each scene's stats dict as JSON.
# Run with the native falcor module (CPython 3.10, packman) from Falcor/media:
#   PYTHONPATH=<Release>/python LD_LIBRARY_PATH=<Release>:<packman python>/lib python3 render-native-scene-stats.py <out.json>
import falcor, sys, json
out = sys.argv[1] if len(sys.argv) > 1 else "scene-stats.json"
# Every media pyscene but the four large production scenes (Bistro, EmeraldSquare, SunTemple, ZeroDay).
scenes = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "test_scenes/cornell_box.pyscene", "test_scenes/geometry_types.pyscene", "test_scenes/material_test.pyscene",
    "Arcade/Arcade.pyscene", "inv_rendering_scenes/bunny_init.pyscene", "inv_rendering_scenes/bunny_ref.pyscene", "inv_rendering_scenes/sphere_init.pyscene",
    "inv_rendering_scenes/spheres_material_init.pyscene", "inv_rendering_scenes/spheres_material_ref.pyscene", "test_scenes/bsdf_optimizer.pyscene",
    "test_scenes/bunny.pyscene", "test_scenes/bunny_war_diff_pt.pyscene", "test_scenes/convergence_test.pyscene", "test_scenes/cornell_box_bunny.pyscene",
    "test_scenes/cornell_box_displaced.pyscene", "test_scenes/nested_dielectrics.pyscene", "test_scenes/smoke.pyscene", "test_scenes/sphere_array.pyscene",
    "test_scenes/tutorial.pyscene", "test_scenes/two_volumes.pyscene", "test_scenes/volume_test.pyscene", "test_scenes/volume_transmittance_test.pyscene",
    "test_scenes/winding_test.pyscene", "DragonBuddha/dragonbuddha.pyscene", "Sponza/sponza.pyscene", "BreakfastRoom/breakfast_room.pyscene",
    "test_scenes/grey_and_white_room/grey_and_white_room.pyscene", "test_scenes/materials/light_leaks.pyscene", "test_scenes/materials/materials.pyscene",
    "test_scenes/tex_lod/spheres_cube.pyscene", "test_scenes/alpha_test/alpha_test.pyscene", "test_scenes/animated_cubes/animated_cubes.pyscene",
    "test_scenes/cesium_man/CesiumMan.pyscene", "test_scenes/curves/two_curves.pyscene",
]
device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
testbed = falcor.Testbed(width=64, height=64, create_window=False, device=device)
# Merge into an existing file: a scene that crashes the process only loses itself (run one scene per process).
try:
    with open(out) as f: result = json.load(f)
except (OSError, ValueError):
    result = {}
for path in scenes:
    try:
        testbed.load_scene(path)
        testbed.frame()
        result[path] = dict(testbed.scene.stats)
        with open(out, "w") as f:
            json.dump(result, f, indent=1, sort_keys=True)
    except Exception as e:
        print(f"{path}: {e}")
with open(out, "w") as f:
    json.dump(result, f, indent=1, sort_keys=True)
