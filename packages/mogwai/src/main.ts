/**
 * Mogwai (web) — the interactive viewer: load a render-graph .py + .pyscene,
 * execute the graph each frame, present the marked output to the canvas.
 */

import { Device, Logger, Profiler, ProgramManager, RenderGraph, ResourceFormat, createPass, encodeExr, initScripting, initSlang, runGraphScript, runSceneScript, runPbrtScene, presentToCanvas, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { CameraController } from "./CameraController.js";
import { buildUIPanel } from "./UIPanel.js";

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
    const scene = url.toLowerCase().endsWith(".pbrt")
        ? await runPbrtScene(state.device, source, baseUrl)
        : await runSceneScript(state.device, source, baseUrl);
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

/**
 * Fetches the Falcor shader tree and wires the program/Slang system onto the
 * device; without it device.programManager is undefined and any pass throws.
 */
async function initProgramSystem(device: Device): Promise<void> {
    const list = (await (await fetch("/packages/falcor/shaders/generated/shader-file-list.json")).json()) as {
        falcorFiles: string[];
        renderPassFiles: string[];
        localFiles: string[];
        externalFiles?: { path: string; url: string }[];
    };
    const sources = new Map<string, string>();
    const missing: string[] = [];
    // Every source path to fetch, tagged with the ProgramManager registry key
    // (Falcor/Source files keep their repo-relative key; local/external the manifest path).
    const jobs: { url: string; key: string }[] = [
        ...list.falcorFiles.map((f) => ({ url: `/Falcor/Source/Falcor/${f}`, key: f })),
        ...list.renderPassFiles.map((f) => ({ url: `/Falcor/Source/${f}`, key: f })),
        ...list.localFiles.map((f) => ({ url: `/packages/falcor/shaders/${f}`, key: f })),
        ...(list.externalFiles ?? []).map(({ path, url }) => ({ url, key: path })),
    ];
    // Bounded concurrency: firing all ~400 fetches at once spikes renderer memory
    // enough to tear the WebGPU context down; a worker pool keeps it flat.
    const CONCURRENCY = 24;
    let next = 0;
    const worker = async () => {
        while (next < jobs.length) {
            const { url, key } = jobs[next++]!;
            const res = await fetch(url);
            if (res.ok) sources.set(key, await res.text());
            else missing.push(`${url} (${res.status})`);
        }
    };
    await Promise.all(Array.from({ length: CONCURRENCY }, worker));
    if (missing.length > 0) {
        Logger.warning(`shader registry: ${missing.length} files failed to fetch; first: ${missing.slice(0, 3).join(", ")}`);
    }
    await initSlang("/tools/slang-wasm/slang-wasm.js");
    device.setProgramManager(new ProgramManager(device, (p) => sources.get(p), [...sources.keys()]));
}

/**
 * Resolves a `?scene=`/`?graph=` param to a fetchable URL: absolute/http(s) pass
 * through; a bare value resolves under /Falcor/media (so `?scene=Arcade/Arcade.pyscene` works).
 */
function resolveAssetUrl(value: string): string {
    return value.startsWith("/") || /^https?:/i.test(value) ? value : `/Falcor/media/${value}`;
}

/**
 * Loads initial content from URL params (`?graph=`/`?script=`, `?scene=`, `?output=`),
 * falling back to the cornell-box path tracer when none are given.
 */
async function loadInitialContent(state: ViewerState, device: Device): Promise<void> {
    const params = new URLSearchParams(location.search);
    const sceneParam = params.get("scene");
    const graphParam = params.get("graph") ?? params.get("script");
    const outputParam = params.get("output");

    if (sceneParam) {
        const url = resolveAssetUrl(sceneParam);
        await loadScene(state, url, url.slice(0, url.lastIndexOf("/")));
    }
    if (graphParam) {
        await loadGraph(state, resolveAssetUrl(graphParam));
    } else if (!sceneParam) {
        // No URL content: default cornell box + the GPU-oracle-verified graph.
        await loadScene(state, "/Falcor/media/test_scenes/cornell_box.pyscene", "/Falcor/media/test_scenes");
        state.graph = buildDefaultGraph(device, canvas.width, canvas.height, state.scene!);
        state.output = state.graph.getOutputNames()[0] ?? null;
    } else {
        // Scene but no graph: run the default path tracer over the chosen scene.
        state.graph = buildDefaultGraph(device, canvas.width, canvas.height, state.scene!);
        state.output = state.graph.getOutputNames()[0] ?? null;
    }
    if (outputParam && state.graph?.getOutputNames().includes(outputParam)) {
        state.output = outputParam;
    }
}

async function main() {
    const device = await Device.create();
    const profiler = new Profiler(device);
    if (profiler.available) device.enableProfiler(profiler);
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get webgpu canvas context");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device.gpuDevice, format });

    await initProgramSystem(device);
    await initScripting("/node_modules/pyodide");

    const state: ViewerState = { device, context, format, graph: null, scene: null, output: null, frame: 0, playing: true };

    // Initial content from URL params (?scene=/?graph=/?output=), or the default
    // cornell-box path tracer when none are given.
    try {
        await loadInitialContent(state, device);
    } catch (e) {
        Logger.warning(`Mogwai: content failed to load (${e})`);
    }

    const passesEl = document.getElementById("passes") as HTMLDivElement;
    const resetAccum = () => {
        (state.graph?.getPass("Accumulate") as { reset?: () => void } | undefined)?.reset?.();
        state.frame = 0;
    };
    const rebuildUI = () => buildUIPanel(passesEl, state.graph, resetAccum);

    wireControls(state, rebuildUI);
    rebuildUI();
    const camControl = new CameraController(canvas);
    (window as unknown as { mogwai: ViewerState }).mogwai = state; // debug/test handle

    let animStart = -1;
    let lastGpuLine = "";
    function frame(now: number) {
        const cam = state.scene?.camera;
        let dirty = cam ? camControl.update(cam, now) : false;
        // Advance scene animation (rebuilds geometry/BVH each frame; no-op if static).
        if (state.playing && state.scene?.isAnimated()) {
            if (animStart < 0) animStart = now;
            if (state.scene.animate((now - animStart) / 1000)) dirty = true;
        }
        if (dirty && state.graph) resetAccum(); // camera or geometry moved: restart accumulation
        if (state.playing && state.graph && state.output) {
            state.graph.execute(device.renderContext);
            const tex = state.graph.getOutput(state.output);
            if (tex) presentToCanvas(device, tex, context!.getCurrentTexture(), format);
            state.frame++;
            const gpu = profiler.available && state.frame % 30 === 0
                ? [...profiler.getStats()].map(([k, v]) => `${k} ${v.toFixed(2)}ms`).join(" · ")
                : null;
            if (gpu) lastGpuLine = gpu;
            status.textContent = `${state.output} · frame ${state.frame}${lastGpuLine ? " · " + lastGpuLine : ""}`;
        }
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

/** Downloads the current marked output (mirrors Mogwai FrameCapture:
 *  float formats save EXR, 8-bit formats save PNG). */
async function captureFrame(state: ViewerState): Promise<void> {
    if (!state.graph || !state.output) return;
    const tex = state.graph.getOutput(state.output);
    if (!tex) return;
    const raw = await state.device.renderContext.readTextureSubresource(tex);
    const name = `${state.output.replace(/\./g, "_")}.${state.frame}`;
    const isFloat = ResourceFormat[tex.format]?.includes("Float") ?? false;
    let blob: Blob;
    let filename: string;
    if (isFloat) {
        const exr = encodeExr(new Float32Array(raw.buffer), tex.width, tex.height);
        blob = new Blob([exr.slice().buffer as ArrayBuffer], { type: "image/x-exr" });
        filename = `${name}.exr`;
    } else {
        // 8-bit path; swizzle BGRA-ordered readbacks to RGBA.
        const bytes = new Uint8ClampedArray(raw.buffer.slice(0) as ArrayBuffer);
        if (ResourceFormat[tex.format]?.startsWith("BGRA")) {
            for (let i = 0; i < bytes.length; i += 4) {
                const b = bytes[i]!;
                bytes[i] = bytes[i + 2]!;
                bytes[i + 2] = b;
            }
        }
        const canvas2 = new OffscreenCanvas(tex.width, tex.height);
        canvas2.getContext("2d")!.putImageData(new ImageData(bytes, tex.width, tex.height), 0, 0);
        blob = await canvas2.convertToBlob({ type: "image/png" });
        filename = `${name}.png`;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

/** Wires the plain-DOM control bar (created in index.html). */
function wireControls(state: ViewerState, rebuildUI: () => void): void {
    const $ = (id: string) => document.getElementById(id);
    ($("capture") as HTMLButtonElement | null)?.addEventListener("click", () => {
        void captureFrame(state);
    });
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
            rebuildUI();
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
