# Native oracle: GBufferRT material channels over tutorial.pyscene, to
# adjudicate the ~6% PT radiance deficit (suspect: normal/specular texture
# slot semantics). normW includes normal mapping; specRough shows the
# sampled specular texture; guideNormalW the guide normal.
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

g = RenderGraph('TutorialGBuffer')
g.addPass(createPass('GBufferRT', {'samplePattern': 'Center'}), 'GBufferRT')
g.markOutput('GBufferRT.normW')
g.markOutput('GBufferRT.specRough')
g.markOutput('GBufferRT.guideNormalW')
g.markOutput('GBufferRT.emissive')
m.addGraph(g)

m.loadScene(os.path.join(root, 'media/test_scenes/tutorial.pyscene'))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.time = 0
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, 'out-native')
m.frameCapture.baseFilename = 'oracle-tutorial-gbuffer'
m.renderFrame()
m.frameCapture.capture()
exit()
