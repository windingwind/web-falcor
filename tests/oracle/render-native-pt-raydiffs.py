# Native oracle: full PathTracer primaryLodMode=RayDiffs over the textured
# tutorial scene (emissive lighting; the checker floor minifies). Renders
# BOTH Mip0 and RayDiffs variants so the web test can prove the mode changes
# sampling before comparing. 64 accumulated frames (NEE decorrelation).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(base, "../../Falcor"))

for mode in ['Mip0', 'RayDiffs']:
    g = RenderGraph('PTRayDiffs' + mode)
    g.addPass(createPass('VBufferRT', {'useAlphaTest': False, 'samplePattern': 'Center'}), 'VBufferRT')
    g.addPass(createPass('PathTracer', {
        'samplesPerPixel': 1,
        'maxSurfaceBounces': 3, 'maxDiffuseBounces': 3, 'maxSpecularBounces': 3, 'maxTransmissionBounces': 10,
        'useRussianRoulette': False,
        'useSER': False,
        'primaryLodMode': mode,
    }), 'PathTracer')
    g.addPass(createPass('AccumulatePass', {'enabled': True, 'precisionMode': 'Single'}), 'Accumulate')
    g.addEdge('VBufferRT.vbuffer', 'PathTracer.vbuffer')
    g.addEdge('PathTracer.color', 'Accumulate.input')
    g.markOutput('Accumulate.output')
    m.addGraph(g)

    m.loadScene(os.path.join(root, 'media/test_scenes/tutorial.pyscene'))
    m.resizeFrameBuffer(256, 256)
    m.ui = False
    m.clock.time = 0
    m.clock.pause()

    m.frameCapture.outputDir = os.path.join(base, 'out-native')
    m.frameCapture.baseFilename = 'oracle-pt-raydiffs-' + mode.lower()
    for _ in range(64):
        m.renderFrame()
    m.frameCapture.capture()
    m.removeGraph(g)
exit()
