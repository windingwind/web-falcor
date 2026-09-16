/**
 * RenderPass.renderUI coverage: every registered pass exposes its native
 * controls through the UIWidgets vocabulary, every control can be flipped
 * without throwing, define-driven options rebuild kernels on the next
 * execute, and scene-bound UIs (BSDFViewer material list) see the scene.
 */

import { RenderGraph, createPass, getRegisteredRenderPasses, initScripting, runSceneScript, type RenderPass, type UIWidgets } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

interface Control {
    kind: string;
    label: string;
    /** Applies a different value through the control's callback. */
    flip: () => void;
}

/** UIWidgets that records controls instead of building DOM. */
class Recorder implements UIWidgets {
    controls: Control[] = [];
    texts: string[] = [];
    text(label: string): void {
        this.texts.push(label);
    }
    button(label: string, onClick: () => void): void {
        this.controls.push({ kind: "button", label, flip: onClick });
    }
    checkbox(label: string, value: boolean, onChange: (v: boolean) => void): void {
        this.controls.push({ kind: "checkbox", label, flip: () => onChange(!value) });
    }
    slider(label: string, value: number, min: number, max: number, _step: number, onChange: (v: number) => void): void {
        this.controls.push({ kind: "slider", label, flip: () => onChange(value === max ? min : max) });
    }
    dropdown(label: string, options: readonly string[], value: string, onChange: (v: string) => void): void {
        const next = options[(Math.max(0, options.indexOf(value)) + 1) % options.length]!;
        this.controls.push({ kind: "dropdown", label, flip: () => onChange(next) });
    }
    group(_label: string): UIWidgets {
        return this; // nested controls land in the same list
    }
    find(label: string): Control | undefined {
        return this.controls.find((c) => c.label === label);
    }
}

const kCreationProps: Record<string, Record<string, unknown>> = {
    ImageLoader: { filename: "test_images/smoke_puff.png" },
};

gpuTest("RenderUI.everyPassExposesFlippableControls", async ({ device }) => {
    const withControls: string[] = [];
    const without: string[] = [];
    for (const type of getRegisteredRenderPasses()) {
        let pass: RenderPass;
        try {
            pass = createPass(device, type, kCreationProps[type] ?? {});
        } catch (e) {
            console.error(`# render-ui: ${type} needs creation props (${String(e).split("\n")[0]}), skipped`);
            continue;
        }
        const rec = new Recorder();
        pass.renderUI(rec);
        if (rec.controls.length === 0) {
            without.push(type);
            continue;
        }
        withControls.push(`${type}(${rec.controls.length})`);
        for (const c of rec.controls) c.flip(); // every control accepts a changed value
        pass.renderUI(new Recorder()); // and the pass still renders its UI afterwards
    }
    console.error(`# render-ui: controls: ${withControls.join(" ")}`);
    console.error(`# render-ui: no controls: ${without.join(" ")}`);
    // Native passes without any renderUI: InvalidPixelDetection, RenderPassTemplate; RTXDIPass's options UI lives in
    // the RTXDI module (⏳); BSDFViewer only lists controls once a scene is set (covered below).
    expectEq(without.every((t) => ["InvalidPixelDetectionPass", "RenderPassTemplate", "RTXDIPass", "BSDFViewer"].includes(t)), true, `unexpected passes without controls: ${without.join(", ")}`);
    expectEq(withControls.length >= 24, true, `${withControls.length} passes expose controls`);
});

gpuTest("RenderUI.controlsDriveOptionsAndKernelRebuild", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const sceneSource = await (await fetch("/Falcor/media/test_scenes/cornell_box.pyscene")).text();
    const scene = await runSceneScript(device, sceneSource, "/Falcor/media/test_scenes");
    scene.camera.setAspectRatio(1);

    // Options round-trip through getProperties (PathTracer had none before).
    const pt = createPass(device, "PathTracer", {});
    const ptUI = new Recorder();
    pt.renderUI(ptUI);
    ptUI.find("Max surface bounces")!.flip(); // -> 254 clamps nothing, then per-lobe limits follow the surface cap
    expectEq(pt.getProperties().get("maxSurfaceBounces", 0), 254, "surface bounce slider writes the static param");
    ptUI.find("Emissive sampler")!.flip();
    expectEq(pt.getProperties().get("emissiveSampler", ""), "Power", "emissive sampler dropdown cycles LightBVH -> Power");

    // BSDFViewer lists the scene's materials once a scene is set.
    const viewer = createPass(device, "BSDFViewer", {});
    const empty = new Recorder();
    viewer.renderUI(empty);
    expectEq(empty.controls.length, 0, "no controls without a scene");
    viewer.setScene(scene);
    const full = new Recorder();
    viewer.renderUI(full);
    expectEq(full.find("Materials") !== undefined, true, "material dropdown present with a scene");

    // Define-driven options rebuild kernels: VBufferRT -> MinimalPathTracer -> ToneMapper graph keeps executing after UI edits.
    const graph = new RenderGraph(device, "UI");
    const vbuf = graph.addPass(createPass(device, "VBufferRT", { samplePattern: "Center" }), "VBufferRT");
    const mpt = graph.addPass(createPass(device, "MinimalPathTracer", { maxBounces: 1 }), "MinimalPathTracer");
    const tm = graph.addPass(createPass(device, "ToneMapper", {}), "ToneMapper");
    graph.addEdge("VBufferRT.vbuffer", "MinimalPathTracer.vbuffer");
    graph.addEdge("MinimalPathTracer.color", "ToneMapper.src");
    graph.markOutput("MinimalPathTracer.color");
    graph.markOutput("ToneMapper.dst");
    graph.onResize(64, 64);
    graph.setScene(scene);
    const ctx = device.renderContext;
    graph.execute(ctx);
    const before = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("MinimalPathTracer.color")!)).buffer);

    const vbUI = new Recorder();
    vbuf.renderUI(vbUI);
    vbUI.find("Sample pattern")!.flip(); // Center -> DirectX (camera jitter generator swapped)
    vbUI.find("Alpha Test")!.flip(); // define -> kernel rebuild
    expectEq(vbuf.getProperties().get("samplePattern", ""), "DirectX", "VBufferRT pattern dropdown");
    const mptUI = new Recorder();
    mpt.renderUI(mptUI);
    mptUI.find("Max bounces")!.flip(); // 1 -> 16, define -> kernel rebuild
    expectEq(mpt.getProperties().get("maxBounces", 0), 16, "MinimalPathTracer bounce slider");
    graph.execute(ctx);
    graph.execute(ctx);
    const after = new Float32Array((await ctx.readTextureSubresource(graph.getOutput("MinimalPathTracer.color")!)).buffer);
    let finite = true;
    let sum = 0;
    let changed = 0;
    for (let i = 0; i < after.length; i += 4) {
        finite &&= Number.isFinite(after[i]!);
        sum += after[i]!;
        if (Math.abs(after[i]! - before[i]!) > 1e-4) changed++;
    }
    expectEq(finite && sum > 0, true, "rebuilt kernels render a finite, non-black frame");
    expectEq(changed > 0, true, `more bounces + jitter change the image (${changed} px)`);

    // Native "Output size" control: the dropdown requests a graph recompile (Default -> Fixed = 512x512 at a 64x64 graph).
    expectEq(graph.getOutput("ToneMapper.dst")!.width, 64, "ToneMapper output follows the graph size by default");
    const tmUI = new Recorder();
    tm.renderUI(tmUI);
    tmUI.find("Output size")!.flip();
    expectEq(tm.getProperties().get("outputSize", ""), "Fixed", "output size dropdown cycles Default -> Fixed");
    graph.execute(ctx);
    expectEq(graph.getOutput("ToneMapper.dst")!.width, 512, "requestRecompile re-reflected the pass at its fixed size");
});
