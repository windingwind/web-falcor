/**
 * Mogwai (web) — the interactive viewer that ties the framework together,
 * mirroring Source/Mogwai's core loop: load a render-graph .py script + a
 * .pyscene, execute the graph each frame, and present a marked output to the
 * canvas. Plain-DOM controls (graph/scene pickers, play/pause, output picker,
 * frame counter) stand in for the native Dear ImGui panels; the ImGui-wasm
 * RenderGraphEditor node UI is a documented stretch (docs §8.3) — it is
 * dev tooling orthogonal to Falcor's rendering parity.
 *
 * The render loop (runGraphScript/runSceneScript/RenderGraph.execute) is the
 * same code the GPU test harness verifies against native; this module adds the
 * swapchain-present path and the browser wiring around it.
 */

import { Device, Logger, RenderGraph, createPass, initScripting, runGraphScript, runSceneScript, presentToCanvas, type Scene } from "@web-falcor/falcor";

const canvas = document.getElementById("canvas") as HTMLCanvasElement;
const status = document.getElementById("status") as HTMLDivElement;

interface ViewerState {
    device: Device;
    context: GPUCanvasContext;
    format: GPUTextureFormat;
    graph: RenderGraph | null;
    scene: Scene | null;
    output: string | null;
    frame: number;
    playing: boolean;
}

async function loadGraph(state: ViewerState, url: string): Promise<void> {
    const source = await (await fetch(url)).text();
    const [graph] = await runGraphScript(state.device, source);
    graph!.onResize(canvas.width, canvas.height);
    if (state.scene) graph!.setScene(state.scene);
    state.graph = graph!;
    state.output = graph!.getOutputNames()[0] ?? null;
    state.frame = 0;
}

async function loadScene(state: ViewerState, url: string, baseUrl: string): Promise<void> {
    const source = await (await fetch(url)).text();
    const scene = await runSceneScript(state.device, source, baseUrl);
    scene.camera.setAspectRatio(canvas.width / canvas.height);
    state.scene = scene;
    if (state.graph) state.graph.setScene(scene);
    state.frame = 0;
}

/** The verified cornell path-tracer graph (matches the GPU oracle setup). */
function buildDefaultGraph(device: Device, width: number, height: number, scene: Scene): RenderGraph {
    const graph = new RenderGraph(device, "Default");
    graph.onResize(width, height);
    graph.addPass(createPass(device, "VBufferRT", { useAlphaTest: false }), "VBufferRT");
    graph.addPass(createPass(device, "PathTracer", { samplesPerPixel: 1, emissiveSampler: "LightBVH" }), "PathTracer");
    graph.addPass(createPass(device, "AccumulatePass", { enabled: true, precisionMode: "Single" }), "Accumulate");
    graph.addPass(createPass(device, "ToneMapper", { autoExposure: false }), "ToneMapper");
    graph.addEdge("VBufferRT.vbuffer", "PathTracer.vbuffer");
    graph.addEdge("PathTracer.color", "Accumulate.input");
    graph.addEdge("Accumulate.output", "ToneMapper.src");
    graph.markOutput("ToneMapper.dst");
    graph.setScene(scene);
    return graph;
}

async function main() {
    const device = await Device.create();
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get webgpu canvas context");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device.gpuDevice, format });

    await initScripting("/node_modules/pyodide");

    const state: ViewerState = { device, context, format, graph: null, scene: null, output: null, frame: 0, playing: true };

    // Default content: a path-traced cornell box (VBufferRT -> PathTracer),
    // the same graph the GPU oracle verifies, so the viewer opens on pixels.
    try {
        await loadScene(state, "/Falcor/media/test_scenes/cornell_box.pyscene", "/Falcor/media/test_scenes");
        state.graph = buildDefaultGraph(device, canvas.width, canvas.height, state.scene!);
        state.output = state.graph.getOutputNames()[0] ?? null;
    } catch (e) {
        Logger.warning(`Mogwai: default content failed to load (${e})`);
    }

    wireControls(state);

    function frame() {
        if (state.playing && state.graph && state.output) {
            state.graph.execute(device.renderContext);
            const tex = state.graph.getOutput(state.output);
            if (tex) presentToCanvas(device, tex, context!.getCurrentTexture(), format);
            state.frame++;
            status.textContent = `${state.output} · frame ${state.frame}`;
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

/** Wires the plain-DOM control bar (created in index.html). */
function wireControls(state: ViewerState): void {
    const $ = (id: string) => document.getElementById(id);
    ($("play") as HTMLButtonElement | null)?.addEventListener("click", () => {
        state.playing = !state.playing;
        ($("play") as HTMLButtonElement).textContent = state.playing ? "Pause" : "Play";
    });
    ($("graphFile") as HTMLInputElement | null)?.addEventListener("change", async (ev) => {
        const file = (ev.target as HTMLInputElement).files?.[0];
        if (file) {
            const [graph] = await runGraphScript(state.device, await file.text());
            graph!.onResize(canvas.width, canvas.height);
            if (state.scene) graph!.setScene(state.scene);
            state.graph = graph!;
            state.output = graph!.getOutputNames()[0] ?? null;
            state.frame = 0;
            refreshOutputs(state);
        }
    });
    refreshOutputs(state);
}

function refreshOutputs(state: ViewerState): void {
    const sel = document.getElementById("output") as HTMLSelectElement | null;
    if (!sel || !state.graph) return;
    sel.innerHTML = "";
    for (const name of state.graph.getOutputNames()) {
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
    }
    if (state.output) sel.value = state.output;
    sel.onchange = () => {
        state.output = sel.value;
    };
}

main().catch((err) => {
    Logger.error(String(err));
    status.textContent = `FAILED: ${err.message ?? err}`;
});
