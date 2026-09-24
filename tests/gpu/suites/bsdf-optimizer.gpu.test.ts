/**
 * BSDFOptimizer over the upstream bsdf_optimizer.pyscene (PBRTDiffuse 0/1, Standard MetalRough
 * 4/5, PBRTConductor 6/7). Native can't serve as an oracle on this machine (its build fails to
 * link the optimizer kernel), so:
 * 1. The backward-mode gradients (compute_bsdf_grads) match central finite differences of the
 *    loss, which is read off the viewer's |f_init - f_ref| viewport on the same slice.
 * 2. Adam steps reduce the loss (native's unclamped Adam state can overshoot individual
 *    parameters, e.g. the Standard base color past 1 under the specular peak's gradients).
 */

import { RenderGraph, createPass, deserializeMaterialParams, initScripting, runSceneScript, serializeMaterialParams } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import type { BSDFOptimizer } from "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const [w, h] = [384, 128];

gpuTest("BSDFOptimizer.gradientsMatchFiniteDifferencesAndConverge", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const ctx = device.renderContext;
    const lines: string[] = [];
    for (const [init, ref, active] of [[0, 1, [0, 1, 2]], [4, 5, [0, 1, 2, 3, 4]], [6, 7, [0, 1, 2, 3, 4, 5, 6, 7]]] as const) {
        const scene = await runSceneScript(device, await (await fetch("/Falcor/media/test_scenes/bsdf_optimizer.pyscene")).text(), "/Falcor/media/test_scenes");
        const graph = new RenderGraph(device, "BSDFOptimizer");
        graph.addPass(createPass(device, "BSDFOptimizer", { initMaterialID: init, refMaterialID: ref }), "BSDFOptimizer");
        graph.markOutput("BSDFOptimizer.output");
        graph.onResize(w, h);
        graph.setScene(scene);
        await graph.init();
        graph.execute(ctx);
        const pass = graph.getPass("BSDFOptimizer") as BSDFOptimizer;
        const extent = pass.bsdf_slice_resolution;

        // Loss from the viewer: 0.5 * |f_init - f_ref|^2 summed over the difference viewport.
        const offset = Math.floor((w - extent * 3) / 2) + extent;
        const yOffset = Math.floor((h - extent) / 2);
        const loss = async () => {
            graph.execute(ctx);
            const px = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("BSDFOptimizer.output")!)).buffer);
            let sum = 0;
            for (let y = 0; y < extent; y++) for (let x = 0; x < extent; x++) for (let c = 0; c < 3; c++) sum += 0.5 * px[((y + yOffset) * w + x + offset) * 4 + c]! ** 2;
            return sum;
        };
        const material = scene.getMaterial(init);
        const base = serializeMaterialParams(material);
        const setParams = (p: Float32Array) => {
            deserializeMaterialParams(material, p);
            scene.updateMaterial(init);
            return serializeMaterialParams(material);
        };
        const gradsBuffer = pass.compute_bsdf_grads();
        const grads = new Float32Array((await ctx.readBuffer(gradsBuffer)).buffer.slice(0, 80));
        let worst = 0;
        const report: string[] = [];
        for (const i of active) {
            const step = 4e-3;
            const plus = base.slice();
            plus[i] += step;
            const minus = base.slice();
            minus[i] -= step;
            // Parameters are stored as float16: use the deltas that actually landed.
            const p = setParams(plus);
            const lp = await loss();
            const m = setParams(minus);
            const lm = await loss();
            const fd = (lp - lm) / (p[i]! - m[i]!);
            const rel = Math.abs(grads[i]! - fd) / Math.max(Math.abs(fd), 1e-2 * Math.max(...Array.from(grads, Math.abs)));
            worst = Math.max(worst, rel);
            report.push(`${i}:${grads[i]!.toPrecision(4)}/${fd.toPrecision(4)}`);
        }
        setParams(base);
        console.error(`# bsdfopt ${init}/${ref} grads(ad/fd) ${report.join(" ")} worst rel ${worst.toExponential(2)}`);
        lines.push(`${init}/${ref} grads(ad/fd) ${report.join(" ")} worst rel ${worst.toExponential(2)}`);
        expectEq(worst < 0.05, true, `materials ${init}/${ref}: autodiff vs finite differences, worst relative error ${worst}`);

        // Adam: the loss shrinks.
        const err = () => active.reduce((s, i) => s + Math.abs(pass.currentParams[i]! - pass.referenceParams[i]!), 0);
        pass.initOptimization();
        const err0 = err();
        const loss0 = await loss();
        pass.runOptimization = true;
        for (let f = 0; f < 200 && pass.runOptimization; f++) {
            graph.execute(ctx);
            await ctx.readTextureSubresource(graph.getOutput("BSDFOptimizer.output")!);
        }
        await new Promise((r) => setTimeout(r, 50));
        const err1 = err();
        pass.runOptimization = false;
        const loss1 = await loss();
        console.error(`# bsdfopt ${init}/${ref} loss ${loss0.toPrecision(4)} -> ${loss1.toPrecision(4)}, param L1 ${err0.toFixed(4)} -> ${err1.toFixed(4)} after ${pass.stepCount} steps; cur ${Array.from(active, (i) => pass.currentParams[i]!.toFixed(3))} ref ${Array.from(active, (i) => pass.referenceParams[i]!.toFixed(3))}`);
        lines.push(`${init}/${ref} loss ${loss0.toPrecision(4)} -> ${loss1.toPrecision(4)} in ${pass.stepCount} steps`);
        expectEq(pass.stepCount > 20 && loss1 < loss0 * 0.5, true, `materials ${init}/${ref}: optimization reduces the loss (${loss0} -> ${loss1})`);
    }
    console.error(`# bsdfopt: ${lines.join("; ")}`);
});
