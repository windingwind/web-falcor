# Native validation of the vdb_to_nvdb.py conversion: loads the generated
# .nvdb through Mogwai's NanoVDB loader (independent, header-only code) and
# compares stats + 500 point samples against the OpenVDB-parse ground truth.
# Run: Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/validate-nvdb.py --headless
from falcor import *
import os, json

base = os.path.dirname(os.path.abspath(__file__))
scene = os.path.join(base, "assets", "smoke-nvdb-validate.pyscene")
open(scene, "w").write(
    "v = GridVolume('smoke')\n"
    f"v.loadGrid(GridVolume.GridSlot.Density, '{os.path.join(base, 'assets', 'smoke.nvdb')}', 'density')\n"
    "sceneBuilder.addGridVolume(v)\n"
    "camera = Camera()\n"
    "camera.position = float3(25, 0, 55)\n"
    "camera.target = float3(0, 25, 0)\n"
    "camera.up = float3(0, 1, 0)\n"
    "sceneBuilder.addCamera(camera)\n")
m.loadScene(scene)
g = m.scene.gridVolumes[0].densityGrid
ref = json.load(open(os.path.join(base, "assets", "smoke-vdb-samples.json")))
assert g.voxelCount == ref["activeVoxels"], (g.voxelCount, ref["activeVoxels"])
bad = 0
for (x, y, z, val, active) in ref["samples"]:
    if abs(g.getValue(int3(x, y, z)) - val) > 1e-7:
        bad += 1
print("VALIDATION:", "PASS" if bad == 0 else f"FAIL ({bad} mismatches)",
      "| voxelCount", g.voxelCount, "| min/max", g.minValue, g.maxValue)
exit()
