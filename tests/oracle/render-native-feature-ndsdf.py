# Native oracle: SDF grids vs native, two scenes x two SceneDebugger modes.
#
# 1) The upstream test_NDSDFGrids.py scene (unmodified NDSDFGrid.pyscene,
#    procedural cheese, NO triangle meshes) in mode 'GeometryID': exact
#    per-pixel footprint compare (sphere-traced hit/miss silhouette).
#    InstanceID is NOT comparable on this scene: the native mesh-less TLAS
#    reports CommittedInstanceID()+1 (verified: single instance, shader sees
#    1, instance-record reads come back all-zero via robust access) -- a
#    local native artifact contradicting native's own fillInstanceDesc
#    asserts. With a mesh present the IDs are correct (probe verified).
# 2) assets/ndsdf-mesh.pyscene (floor quad + the same cheese) in mode
#    'InstanceID': verifies instance ordering (mesh 0, SDF 1) and
#    triangle-BVH vs SDF closest-hit competition.
#
# The default FaceNormal mode is also NOT comparable on this machine: native
# NDSDF gradients come out NaN (FlatShaded probe shows uniform 0.2 ambient;
# numeric gradient reads corners through SPIR-V offset texel fetches) -- the
# web render shows geometrically consistent normals (DESIGN.md 6.3 artifact
# class).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

def render(mode, scene_path, name):
    g = RenderGraph('SceneDebugger')
    SceneDebugger = createPass('SceneDebugger', {'mode': mode})
    g.addPass(SceneDebugger, 'SceneDebugger')
    g.markOutput('SceneDebugger.output')
    m.addGraph(g)
    m.loadScene(scene_path)
    m.resizeFrameBuffer(640, 360)
    m.ui = False
    m.clock.framerate = 60
    m.clock.time = 0
    m.clock.pause()
    m.frameCapture.outputDir = os.path.join(base, "out-native")
    m.frameCapture.baseFilename = name
    frame = 0
    while frame < 64:
        frame += 1
        m.clock.frame = frame
        m.renderFrame()
    m.frameCapture.capture()
    m.removeGraph(g)

render('GeometryID', os.path.join(root, "tests/image_tests/scene/scenes/NDSDFGrid.pyscene"), "oracle-feature-ndsdf")
render('InstanceID', os.path.join(base, "assets/ndsdf-mesh.pyscene"), "oracle-feature-ndsdf-mesh")
exit()
