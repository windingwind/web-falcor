# Native oracle for scene contents: every material's Python properties and texture slots, the analytic lights and
# the cameras of a media scene, as JSON. Run one scene per process from Falcor/media with the native falcor module:
#   PYTHONPATH=<Release>/python LD_LIBRARY_PATH=<Release>:<packman python>/lib python3 render-native-scene-content.py <out.json> <scene.pyscene>
import falcor, sys, json, math
out, path = sys.argv[1], sys.argv[2]

def plain(v):
    if isinstance(v, float) and not math.isfinite(v): return str(v)  # JSON has no inf/nan
    if isinstance(v, (bool, int, float, str)) or v is None: return v
    for n in (4, 3, 2):
        try: return [plain(float(getattr(v, c))) for c in "xyzw"[:n]]
        except AttributeError: pass
    if hasattr(v, "name") and hasattr(v, "value"): return v.name
    return str(v)

kMaterialProps = ["type", "doubleSided", "thinSurface", "alphaMode", "alphaThreshold", "nestedPriority", "IoR", "baseColor", "specularParams",
                  "roughness", "metallic", "transmissionColor", "diffuseTransmission", "specularTransmission", "volumeAbsorption", "volumeScattering",
                  "volumeAnisotropy", "emissiveColor", "emissiveFactor", "displacementScale", "displacementOffset", "indexOfRefraction", "shadingModel"]
kLightProps = ["name", "type", "active", "intensity", "position", "direction", "openingAngle", "penumbraAngle", "angle"]
kCameraProps = ["name", "position", "target", "up", "focalLength", "frameHeight", "focalDistance", "apertureRadius", "shutterSpeed", "ISOSpeed", "nearPlane", "farPlane"]

device = falcor.Device(type=falcor.DeviceType.Vulkan, gpu=0, enable_debug_layer=False)
testbed = falcor.Testbed(width=64, height=64, create_window=False, device=device)
# The real-time clock would advance by the load duration; paused, animated cameras are posed at t = 0.
testbed.clock.pause()
testbed.load_scene(path)
testbed.frame()
scene = testbed.scene
def props(obj, names):
    d = {}
    for k in names:
        try: d[k] = plain(getattr(obj, k))
        except Exception: pass
    return d
materials = []
for m in scene.materials:
    d = {"name": m.name, **props(m, kMaterialProps), "textures": {}}
    for slot in falcor.MaterialTextureSlot.__members__.values():
        try: t = m.getTexture(slot)
        except Exception: t = None
        if t is not None: d["textures"][slot.name] = [t.width, t.height]
    materials.append(d)
result = {"materials": materials, "lights": [props(l, kLightProps) for l in scene.lights], "cameras": [props(c, kCameraProps) for c in scene.cameras]}
try:
    with open(out) as f: all_ = json.load(f)
except (OSError, ValueError):
    all_ = {}
all_[path] = result
with open(out, "w") as f: json.dump(all_, f, indent=1, sort_keys=True)
