# Native oracle: GBufferRT texture-LOD ray cones over the textured tutorial
# scene. Renders BOTH Mip0 and RayCones variants so the test can prove the
# mode actually changes sampling (non-vacuous) before comparing.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

for mode in ['Mip0', 'RayCones']:
    g = RenderGraph('TexLOD' + mode)
    g.addPass(createPass('GBufferRT', {'samplePattern': 'Center', 'texLOD': mode}), 'GBufferRT')
    g.markOutput('GBufferRT.diffuseOpacity')
    m.addGraph(g)

    m.loadScene(os.path.join(root, 'media/test_scenes/tutorial.pyscene'))
    m.resizeFrameBuffer(256, 256)
    m.ui = False
    m.clock.time = 0
    m.clock.pause()

    m.frameCapture.outputDir = os.path.join(base, 'out-native')
    m.frameCapture.baseFilename = 'oracle-texlod-' + mode.lower()
    m.renderFrame()
    m.frameCapture.capture()
    m.removeGraph(g)
exit()
