# Native oracle: SVGF feature graph (see graphs/svgf-feature.py) over
# sphere_array.pyscene, 4 frames (temporal reprojection + history accumulate
# over per-frame path-tracer noise). Captures frame 4.
# sphere_array (not cornell): GBufferRT's linearZ derivative helper
# (computeDdxPosW) hits normalize(0) UB on surfaces whose normal is parallel
# to cameraU/V — cornell's axis-aligned walls make HALF the image garbage
# (native writes z=Inf there!), diverging across compiler stacks. On spheres
# the degenerate set is ~a point per sphere.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

exec(open(os.path.join(base, "graphs/svgf-feature.py")).read())
# Input captures for input-vs-kernel bisection in the web test.
for out in ["PathTracer.color", "PathTracer.albedo", "GBufferRT.linearZ",
            "GBufferRT.guideNormalW", "GBufferRT.emissive", "GBufferRT.mvec"]:
    SVGFFeature.markOutput(out)

m.loadScene(os.path.join(root, "media/test_scenes/sphere_array.pyscene"))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, "out-native")
m.frameCapture.baseFilename = "oracle-feature-svgf"

for i in range(4):
    m.renderFrame()
m.frameCapture.capture()
exit()
