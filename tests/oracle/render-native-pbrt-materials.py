# Native oracle: PBRT material classes via the PBRTImporter:usePBRTMaterials
# Settings option (m.addOptions mirrors the web getGlobalSettings().addOptions).
# Graph replicates the web test: VBufferRT -> PathTracer(spp 1) -> Accumulate,
# 256 accumulated frames (dielectric/transmission variance) (area-light NEE decorrelates single frames).
from falcor import *
import os

base = os.path.dirname(os.path.abspath(__file__))

m.addOptions({'PBRTImporter': {'usePBRTMaterials': True}})

g = RenderGraph('PbrtMaterials')
g.addPass(createPass('VBufferRT', {'useAlphaTest': False}), 'VBufferRT')
g.addPass(createPass('PathTracer', {'samplesPerPixel': 1}), 'PathTracer')
g.addPass(createPass('AccumulatePass', {'enabled': True, 'precisionMode': 'Single'}), 'Accumulate')
g.addEdge('VBufferRT.vbuffer', 'PathTracer.vbuffer')
g.addEdge('PathTracer.color', 'Accumulate.input')
g.markOutput('Accumulate.output')
m.addGraph(g)

m.loadScene(os.path.join(base, 'assets/oracle-pbrt-materials.pbrt'))
m.resizeFrameBuffer(256, 256)
m.ui = False
m.clock.pause()

m.frameCapture.outputDir = os.path.join(base, 'out-native')
m.frameCapture.baseFilename = 'oracle-pbrt-materials'

for _ in range(256):
    m.renderFrame()
m.frameCapture.capture()
exit()
