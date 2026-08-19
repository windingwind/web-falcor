from falcor import *
import os
base = os.path.dirname(os.path.abspath(__file__))
g = RenderGraph('BunnyMask')
g.addPass(createPass('VBufferRT', {'useAlphaTest': False, 'samplePattern': 'Center'}), 'VBufferRT')
g.markOutput('VBufferRT.mask')
g.markOutput('VBufferRT.depth')
m.addGraph(g)
m.loadScene(os.path.join(base, 'assets/oracle-bunny.pyscene'))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.pause()
m.frameCapture.outputDir = os.path.join(base, 'out-native')
m.frameCapture.baseFilename = 'oracle-bunny-mask'
m.renderFrame()
m.frameCapture.capture()
exit()
