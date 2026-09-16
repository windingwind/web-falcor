/**
 * FEATURE VERIFY — texture-LOD ray cones vs native. GBufferRT with
 * texLOD=RayCones over the textured tutorial scene: the material texture
 * arrays now carry real mip chains, so cone-driven LOD selection changes the
 * sampled albedo where textures minify. Non-vacuousness is pinned by also
 * diffing against the native Mip0 capture (28k pixels differ natively).
 *
 * Regenerate the oracles with:
 *   xvfb-run -a Falcor/build/linux-gcc/bin/Debug/Mogwai --script tests/oracle/render-native-texlod-raycones.py --headless
 */

import { initScripting, runGraphScript, runSceneScript } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("TexLODRayCones.matchesNativeOracle", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const graphSource = `
from falcor import *
g = RenderGraph('TexLODRayCones')
g.addPass(createPass('GBufferRT', {'samplePattern': 'Center', 'texLOD': 'RayCones'}), 'GBufferRT')
g.markOutput('GBufferRT.diffuseOpacity')
try: m.addGraph(g)
except NameError: None
`;
    const [graph] = await runGraphScript(device, graphSource);
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/tutorial.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1.0);
    graph!.onResize(size, size);
    graph!.setScene(scene);
    const ctx = device.renderContext;
    graph!.execute(ctx);
    const web = new Float32Array((await ctx.readTextureSubresource(graph!.getOutput("GBufferRT.diffuseOpacity")!)).buffer);

    const load = async (name: string) => {
        const res = await fetch(`/tests/oracle/out-native/${name}.GBufferRT.diffuseOpacity.0.exr`);
        return parseExr(await res.arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
    };
    const cones = await load("oracle-texlod-raycones");
    const mip0 = await load("oracle-texlod-mip0");
    expectEq(cones.width, size, "oracle resolution");

    const diff = (nat: { data: Float32Array; height: number; width: number }) => {
        let sum = 0;
        let bad = 0;
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const wi = (y * size + x) * 4;
                const ni = ((nat.height - 1 - y) * nat.width + x) * 4;
                let pm = 0;
                for (let c = 0; c < 3; c++) {
                    const d = Math.abs(web[wi + c]! - nat.data[ni + c]!);
                    sum += d;
                    pm = Math.max(pm, d);
                }
                if (pm > 1e-2) bad++;
            }
        }
        return { mean: sum / (size * size * 3), bad };
    };
    const vsCones = diff(cones);
    const vsMip0 = diff(mip0);
    console.error(`# texlod-raycones: vsRayCones mean=${vsCones.mean.toExponential(2)} bad=${vsCones.bad}; vsMip0 mean=${vsMip0.mean.toExponential(2)} bad=${vsMip0.bad}`);
    // The web render must match the RayCones capture and clearly NOT the Mip0
    // one. Anisotropic/mip filtering details are implementation-defined across
    // APIs (docs §9 float determinism), so the gates are separation-based with
    // absolute bounds (measured: vsCones 4.8e-3/891 vs vsMip0 1.1e-2/10572).
    expectEq(vsCones.mean < 6e-3, true, `raycones mean ${vsCones.mean}`);
    expectEq(vsCones.bad <= 1500, true, `raycones bad px ${vsCones.bad}`);
    expectEq(vsCones.mean < 0.5 * vsMip0.mean, true, `closer to RayCones than Mip0 (${vsCones.mean} vs ${vsMip0.mean})`);
    expectEq(vsCones.bad * 5 < vsMip0.bad, true, `bad-px separation (${vsCones.bad} vs ${vsMip0.bad})`);
});
