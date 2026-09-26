# Default property dictionaries of every render pass the web registers (plus a few non-default ones),
# as native getDictionary() returns them, in key order. Run by Mogwai: tests/oracle/out-native/pass-properties.json.
from falcor import *
import json, sys

def plain(v):
    if isinstance(v, (bool, int, float, str)) or v is None: return v
    if isinstance(v, dict): return {k: plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [plain(x) for x in v]
    for n in (4, 3, 2):
        try: return [plain(getattr(v, c)) for c in "xyzw"[:n]]
        except AttributeError: pass
    return str(v)

cases = [(n, {}) for n in "AccumulatePass BlitPass BSDFOptimizer BSDFViewer ColorMapPass Composite CrossFade ErrorMeasurePass FLIPPass GaussianBlur GBufferRaster GBufferRT ImageLoader InvalidPixelDetectionPass MinimalPathTracer ModulateIllumination NRD OverlaySamplePass PathTracer PixelInspectorPass RenderPassTemplate RTXDIPass SceneDebugger SDFEditor SideBySidePass SimplePostFX SplitScreenPass SVGFPass TAA TestRtProgram ToneMapper VBufferRaster VBufferRT WARDiffPathTracer WhittedRayTracer".split()]
cases += [("PathTracer", {'fixedSeed': 3, 'misHeuristic': 'PowerExp', 'outputSize': 'Fixed', 'fixedOutputSize': uint2(64, 32), 'colorFormat': 'RGBA32F'})]
out = {}
for i, (name, props) in enumerate(cases):
    key = name if not props else f"{name}#{i}"
    try:
        out[key] = {"props": plain(dict(props)), "dict": plain(dict(createPass(name, props).getDictionary()))}
    except Exception as e:
        out[key] = {"props": plain(dict(props)), "error": str(e)}
path = sys.argv[1] if len(sys.argv) > 1 else "pass-properties.json"
with open(path, "w") as f: json.dump(out, f, indent=1)
exit()
