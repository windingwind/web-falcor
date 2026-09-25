/**
 * Mogwai (web) — the interactive viewer: load a render-graph .py + .pyscene,
 * execute the graph each frame, present the marked output to the canvas.
 */

import { FrameCaptureExtension, captureOutput } from "./FrameCapture.js";
import { AssetCategory, AssetResolver, isAbsoluteUrl, kProjectMediaUrl, Clock, Device, Logger, Profiler, ProfilerUI, VideoRecorder, ProgramManager, RenderGraph, ResourceFormat, Bitmap, BitmapExportFlags, createPass, initScripting, initSlang, runConsoleCommand, runGraphScript, runSceneScript, nativeKeyCode, type MogwaiCallbacks, runPbrtScene, runMitsubaScene, presentToCanvas, OverlayDrawList, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { CameraController, kCameraControllerTypes, kUpDirectionNames } from "./CameraController.js";
import { buildUIPanel } from "./UIPanel.js";
import { GraphEditor } from "./GraphEditor.js";
import { saveConfig } from "./SaveConfig.js";

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
    /** Global time control (mirrors m.clock; drives scene animation). */
    clock: Clock;
    timingCapture: TimingCapture;
    /** m.frameCapture (Mogwai's FrameCapture extension). */
    frameCapture: FrameCaptureExtension | null;
    /** Scene panel "Animate Scene" (mirrors AnimationController::setEnabled). */
    animateScene: boolean;
    /** Last graph execute error (graph edits can leave inputs unconnected); cleared on the next edit. */
    graphError: string | null;
    /** The loaded scene's path (Save Config's m.loadScene argument). */
    scenePath: string | null;
    /** m.sceneUpdateCallback / m.keyCallback, set from graph scripts or the console. */
    callbacks: MogwaiCallbacks;
}

/** Mirrors the Mogwai TimingCapture extension. Web divergence (docs §9):
 *  no file IO — captureFrameTime(name) starts collecting per-frame CPU ms,
 *  calling it again (or with no name) stops and downloads the log. */
class TimingCapture {
    private times: number[] = [];
    private active = false;
    private filename = "timing.txt";

    captureFrameTime(filename?: string): string {
        if (!this.active && filename) {
            this.filename = filename;
            this.times = [];
            this.active = true;
            return `capturing frame times to ${filename}`;
        }
        this.active = false;
        const blob = new Blob([this.times.map((t) => t.toFixed(3)).join("\n") + "\n"], { type: "text/plain" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = this.filename;
        a.click();
        URL.revokeObjectURL(a.href);
        return `downloaded ${this.times.length} frame times as ${this.filename}`;
    }

    record(ms: number): void {
        if (this.active) this.times.push(ms);
    }
}

async function loadGraph(state: ViewerState, url: string): Promise<void> {
    const source = await (await fetch(url)).text();
    const [graph] = await runGraphScript(state.device, source, { frameCapture: state.frameCapture }, state.callbacks);
    await graph!.init(); // async pass initialization (ImageLoader etc.; docs §9)
    graph!.onResize(canvas.width, canvas.height);
    if (state.scene) graph!.setScene(state.scene);
    state.graph = graph!;
    state.output = graph!.getOutputNames()[0] ?? null;
    state.frame = 0;
}

async function loadScene(state: ViewerState, url: string, baseUrl: string): Promise<void> {
    const source = await (await fetch(url)).text();
    const lower = url.toLowerCase();
    const scene = lower.endsWith(".pbrt")
        ? await runPbrtScene(state.device, source, baseUrl)
        : lower.endsWith(".xml") // Mitsuba scenes are the only .xml we load
          ? await runMitsubaScene(state.device, source, baseUrl)
          : await runSceneScript(state.device, source, baseUrl, { cache: true, path: url }); // OPFS scene cache: fast reloads
    if (scene.importPaths[0] !== url) scene.importPaths.unshift(url);
    scene.camera.setAspectRatio(canvas.width / canvas.height);
    state.scene = scene;
    state.scenePath = url;
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
    const sources = await fetchShaderSources();
    await initSlang("/tools/slang-wasm/slang-wasm.js");
    device.setProgramManager(new ProgramManager(device, (p) => sources.get(p), [...sources.keys()]));
}

/**
 * Re-fetches every shader and recompiles all programs in place (native F5).
 * The scene, the camera and the render graph stay as they are; passes pick the
 * new kernels up on their next frame.
 */
async function reloadShaders(device: Device): Promise<void> {
    const sources = await fetchShaderSources();
    device.programManager.reloadAllPrograms({ resolveSource: (p) => sources.get(p), filePaths: [...sources.keys()] });
    Logger.info("Shaders reloaded.");
}

/** Fetches the Falcor shader tree into a path -> source map. */
async function fetchShaderSources(): Promise<Map<string, string>> {
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
    return sources;
}

/**
 * Resolves a `?scene=`/`?graph=` param to a fetchable URL: absolute/http(s) pass
 * through; a bare value resolves under /Falcor/media (so `?scene=Arcade/Arcade.pyscene` works).
 */
async function resolveAssetUrl(value: string): Promise<string> {
    if (isAbsoluteUrl(value)) return value;
    return (await AssetResolver.getDefaultResolver().resolvePath(value, AssetCategory.Scene)) || `${kProjectMediaUrl}/${value}`;
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
        const url = await resolveAssetUrl(sceneParam);
        await loadScene(state, url, url.slice(0, url.lastIndexOf("/")));
    }
    if (graphParam) {
        await loadGraph(state, await resolveAssetUrl(graphParam));
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

const videoRecorder = new VideoRecorder();

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

    const state: ViewerState = { device, context, format, graph: null, scene: null, output: null, frame: 0, playing: true, clock: new Clock(), timingCapture: new TimingCapture(), frameCapture: null, animateScene: true, graphError: null, scenePath: null, callbacks: { sceneUpdateCallback: null, keyCallback: null } };
    /** A failing python callback is logged and dropped (it would otherwise fail every frame). */
    const runRendererCallback = (key: keyof MogwaiCallbacks, fn: () => unknown): unknown => {
        try {
            return fn();
        } catch (e) {
            Logger.error(`m.${key} failed and was removed: ${String(e).split("\n").slice(-2).join(" ")}`);
            state.callbacks[key] = null;
            return false;
        }
    };
    // Renderer::onKeyEvent: m.keyCallback(pressed, key) sees presses/releases first; True consumes them.
    const onScriptKey = (ev: KeyboardEvent) => {
        const cb = state.callbacks.keyCallback;
        if (!cb || ev.repeat || ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
        if (runRendererCallback("keyCallback", () => cb(ev.type === "keydown", nativeKeyCode(ev.code)))) {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        }
    };
    window.addEventListener("keydown", onScriptKey, true);
    window.addEventListener("keyup", onScriptKey, true);
    state.frameCapture = new FrameCaptureExtension(device, () => state.graph, (name) => (state.graph?.name === name ? state.graph : null), () => state.clock.getFrame());

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
    const camControl = new CameraController(canvas);
    // Mirrors Scene::setCameraControlsEnabled (e.g. the SDF editor takes the mouse while a modifier is held).
    camControl.inputEnabled = () => state.scene?.cameraControlsEnabled ?? true;
    const rebuildUI = () =>
        buildUIPanel(passesEl, state.graph, resetAccum, state.scene, {
            notify: resetAccum,
            rebuild: () => {
                resetAccum();
                rebuildUI();
            },
            getAnimate: () => state.animateScene,
            setAnimate: (v) => (state.animateScene = v),
            cameraControl: {
                types: kCameraControllerTypes,
                getType: () => camControl.getControllerType(),
                setType: (v) => camControl.setControllerType(v as (typeof kCameraControllerTypes)[number]),
                upNames: kUpDirectionNames,
                getUp: () => camControl.getUpDirection() as number,
                setUp: (i) => camControl.setUpDirection(i),
                getSpeed: () => camControl.getSpeed(),
                setSpeed: (v) => {
                    camControl.setSpeed(v);
                    if (state.scene) state.scene.cameraSpeed = camControl.getSpeed();
                },
            },
        });

    wireControls(state, rebuildUI);
    wireConsole(state, resetAccum, rebuildUI, profiler);
    wireMouseForwarding(state);
    wireGraphEditor(state, rebuildUI, resetAccum);
    wirePixelPicking(state, rebuildUI);
    rebuildUI();
    // Profiler panel (native: P toggles the profiler window).
    const profilerPanel = document.getElementById("profiler") as HTMLDivElement;
    const overlayCanvas = document.getElementById("overlay") as HTMLCanvasElement;
    const overlay = new OverlayDrawList(overlayCanvas.getContext("2d")!);
    // Mirrors MogwaiSettings' mShowOverlayUI: the graph's passes draw over each presented frame.
    const renderOverlay = () => {
        if (overlayCanvas.hidden) return;
        if (overlayCanvas.width !== canvas.width || overlayCanvas.height !== canvas.height) [overlayCanvas.width, overlayCanvas.height] = [canvas.width, canvas.height];
        overlay.ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        state.graph?.renderOverlayUI(overlay);
    };
    const profilerUI = new ProfilerUI(profiler, profilerPanel);
    window.addEventListener("keydown", (ev) => {
        if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
        if (ev.key === "p" || ev.key === "P") profilerPanel.hidden = !profilerPanel.hidden;
        // Reload shaders in place. Native binds this to F5, which the browser
        // owns, so the viewer uses F6 (docs §9).
        const modified = ev.ctrlKey || ev.shiftKey || ev.altKey;
        // Scene::onKeyEvent: F3 adds a viewpoint; C or F7 with a modifier re-enables camera animation.
        if (ev.key === "F3" && !modified) {
            ev.preventDefault();
            state.scene?.addViewpoint();
            rebuildUI();
        } else if (modified && (ev.key === "c" || ev.key === "C" || ev.key === "F7")) {
            ev.preventDefault();
            if (state.scene) state.scene.camera.animated = true;
        } else if (ev.key === "F7") {
            ev.preventDefault();
            overlayCanvas.hidden = !overlayCanvas.hidden;
        }
        if (ev.key === "F6") {
            ev.preventDefault();
            void reloadShaders(state.device).then(() => (state.frame = 0));
        }
    });
    (window as unknown as { mogwaiProfiler: { profiler: Profiler; ui: ProfilerUI } }).mogwaiProfiler = { profiler, ui: profilerUI };
    const pixelZoom = wirePixelZoom();
    (window as unknown as { mogwaiPixelZoom: PixelZoom }).mogwaiPixelZoom = pixelZoom;
    (window as unknown as { mogwai: ViewerState }).mogwai = state; // debug/test handle
    (window as unknown as { mogwaiCamControl: CameraController }).mogwaiCamControl = camControl;

    let lastGpuLine = "";
    let lastNow = -1;
    function frame(now: number) {
        const cam = state.scene?.camera;
        // Renderer::onFrameRender: m.sceneUpdateCallback, then Scene::update (the scene's python updateCallback first).
        if (state.graph) runRendererCallback("sceneUpdateCallback", () => state.callbacks.sceneUpdateCallback?.(state.scene, state.clock.getTime()));
        state.scene?.runUpdateCallback(state.clock.getTime());
        let dirty = cam ? camControl.update(cam, now, state.scene ?? undefined) : false;
        // Scene::onKeyEvent: moving the camera by hand stops its animation.
        if (dirty && cam) cam.animated = false;
        // Advance the global clock (mirrors m.clock; console pause/frame stepping applies here).
        if (state.playing) {
            state.clock.tick();
            // Scene animation follows clock time (rebuilds geometry/BVH; no-op if static).
            if (state.animateScene && state.scene?.isAnimated() && state.scene.animate(state.clock.getTime())) dirty = true;
            // Grid-volume sequences play back on the same clock (Scene::updateGridVolumes).
            if (state.animateScene && state.scene?.updateGridVolumePlayback(state.clock.getTime())) dirty = true;
        }
        if (dirty && state.graph) resetAccum(); // camera or geometry moved: restart accumulation
        if (state.playing && state.graph && state.output) {
            if (lastNow >= 0) state.timingCapture.record(now - lastNow);
            lastNow = now;
            state.frameCapture?.beginFrame();
            try {
                state.graph.execute(device.renderContext);
            } catch (e) {
                // Mid-edit graphs (unconnected required inputs) must not kill the frame loop.
                state.graphError = String(e);
            }
            const tex = state.graph.getOutput(state.output);
            if (tex) presentToCanvas(device, tex, context!.getCurrentTexture(), format);
            renderOverlay();
            void state.frameCapture?.endFrame();
            pixelZoom.render();
            if (videoRecorder.recording) videoRecorder.captureFrame();
            state.frame++;
            const gpu = profiler.available && state.frame % 30 === 0
                ? [...profiler.getStats()].map(([k, v]) => `${k} ${v.toFixed(2)}ms`).join(" · ")
                : null;
            if (gpu) lastGpuLine = gpu;
            status.textContent = state.graphError ? `graph error: ${state.graphError}` : `${state.output} · frame ${state.frame}${lastGpuLine ? " · " + lastGpuLine : ""}`;
        }
        if (!profilerPanel.hidden) profilerUI.render();
        requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
}

/** Downloads the current marked output, one file per marked channel mask (mirrors Mogwai FrameCapture). */
async function captureFrame(state: ViewerState): Promise<void> {
    if (!state.graph || !state.output) return;
    const index = state.graph.getOutputNames().indexOf(state.output);
    await captureOutput(state.device, state.graph, index, `${state.output.replace(/\./g, "_")}.${state.frame}`);
}

/** Interactive python console (mirrors Mogwai's console; Enter runs the line). */
function wireConsole(state: ViewerState, resetAccum: () => void, rebuildUI: () => void, profiler: Profiler): void {
    const panel = document.getElementById("console") as HTMLDivElement | null;
    const log = document.getElementById("consoleLog") as HTMLDivElement | null;
    const input = document.getElementById("consoleInput") as HTMLInputElement | null;
    const toggle = document.getElementById("consoleToggle") as HTMLButtonElement | null;
    if (!panel || !log || !input || !toggle) return;
    toggle.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        if (!panel.hidden) input.focus();
    });
    const append = (text: string, cls?: string) => {
        const div = document.createElement("div");
        if (cls) div.className = cls;
        div.textContent = text;
        log.appendChild(div);
        log.scrollTop = log.scrollHeight;
    };
    const history: string[] = [];
    let histIdx = 0;
    input.addEventListener("keydown", (ev) => {
        ev.stopPropagation(); // keep WASD etc. out of the camera controller
        if (ev.key === "ArrowUp" && history.length > 0) {
            histIdx = Math.max(0, histIdx - 1);
            input.value = history[histIdx] ?? "";
        } else if (ev.key === "ArrowDown") {
            histIdx = Math.min(history.length, histIdx + 1);
            input.value = history[histIdx] ?? "";
        } else if (ev.key === "Enter" && input.value.trim()) {
            const src = input.value;
            history.push(src);
            histIdx = history.length;
            input.value = "";
            append(`>>> ${src}`, "in");
            try {
                const out = runConsoleCommand(state.device, src, { scene: state.scene, graph: state.graph, clock: state.clock, timingCapture: state.timingCapture, frameCapture: state.frameCapture, profiler, callbacks: state.callbacks });
                if (out) append(out);
            } catch (e) {
                append(String(e), "err");
            }
            // Edits likely changed scene/pass state: restart accumulation, refresh panels.
            resetAccum();
            refreshOutputs(state);
            rebuildUI();
        }
    });
}

/** Canvas click (without drag) selects the pixel on picking-capable passes
 *  (PixelInspectorPass.setCursorPosition), then refreshes the pass panel once
 *  the async readback has landed. */
function wirePixelPicking(state: ViewerState, rebuildUI: () => void): void {
    let downPos: [number, number] | null = null;
    canvas.addEventListener("mousedown", (ev) => (downPos = [ev.clientX, ev.clientY]));
    canvas.addEventListener("click", (ev) => {
        if (!downPos || Math.hypot(ev.clientX - downPos[0], ev.clientY - downPos[1]) > 3 || !state.graph) return;
        const rect = canvas.getBoundingClientRect();
        const nx = (ev.clientX - rect.left) / rect.width;
        const ny = (ev.clientY - rect.top) / rect.height;
        let any = false;
        for (const { pass } of state.graph.getPasses()) {
            const p = pass as { setCursorPosition?: (x: number, y: number) => void };
            if (typeof p.setCursorPosition === "function") {
                p.setCursorPosition(nx, ny);
                any = true;
            }
        }
        if (any) setTimeout(rebuildUI, 250); // async pixel-data readback lands ~1 frame later
    });
}

/** Mirrors Utils/UI/PixelZoom: hold Z to magnify the pixels under the cursor (wheel = zoom). */
interface PixelZoom {
    render(): void;
    active: boolean;
    srcZoomSize: number;
}

/**
 * Native PixelZoom copies an mSrcZoomSize² block around the cursor into a 200² point-
 * filtered window centred on the cursor (clamped to the edges) while Z is held; the
 * wheel grows/shrinks the source block by 4 px (min 3). Web: a 2D overlay canvas drawn
 * from the WebGPU canvas after each present.
 */
function wirePixelZoom(): PixelZoom {
    const kDstZoomSize = 200;
    const kZoomCoefficient = 4;
    const overlay = document.getElementById("zoom") as HTMLCanvasElement;
    overlay.width = kDstZoomSize;
    overlay.height = kDstZoomSize;
    const c2d = overlay.getContext("2d")!;
    c2d.imageSmoothingEnabled = false;
    const zoom: PixelZoom = { active: false, srcZoomSize: 5, render: () => {} };
    let mouse: [number, number] = [0.5, 0.5]; // normalized canvas position
    window.addEventListener("keydown", (ev) => {
        if ((ev.key === "z" || ev.key === "Z") && !(ev.target instanceof HTMLInputElement) && !(ev.target instanceof HTMLTextAreaElement)) zoom.active = true;
    });
    window.addEventListener("keyup", (ev) => {
        if (ev.key === "z" || ev.key === "Z") zoom.active = false;
    });
    window.addEventListener("blur", () => (zoom.active = false));
    canvas.addEventListener("mousemove", (ev) => {
        const rect = canvas.getBoundingClientRect();
        mouse = [(ev.clientX - rect.left) / rect.width, (ev.clientY - rect.top) / rect.height];
    });
    canvas.addEventListener(
        "wheel",
        (ev) => {
            if (!zoom.active) return;
            zoom.srcZoomSize = Math.max(zoom.srcZoomSize + kZoomCoefficient * Math.sign(ev.deltaY), 3);
            ev.preventDefault();
            ev.stopImmediatePropagation();
        },
        { capture: true, passive: false },
    );
    zoom.render = () => {
        overlay.hidden = !zoom.active;
        if (!zoom.active) return;
        const rect = canvas.getBoundingClientRect();
        const offset = Math.floor(zoom.srcZoomSize / 2);
        // Source block in canvas pixels, clamped so it stays inside the framebuffer.
        const clampEdge = (v: number, size: number, off: number) => Math.min(Math.max(v, off), size - off);
        const sx = clampEdge(mouse[0] * canvas.width, canvas.width, offset);
        const sy = clampEdge(mouse[1] * canvas.height, canvas.height, offset);
        c2d.clearRect(0, 0, kDstZoomSize, kDstZoomSize);
        c2d.imageSmoothingEnabled = false;
        c2d.drawImage(canvas, sx - offset, sy - offset, 2 * offset, 2 * offset, 0, 0, kDstZoomSize, kDstZoomSize);
        // Window centred on the cursor (CSS px), clamped to the canvas rectangle like native.
        const half = kDstZoomSize / 2;
        const cx = clampEdge(mouse[0] * rect.width, rect.width, half);
        const cy = clampEdge(mouse[1] * rect.height, rect.height, half);
        overlay.style.left = `${rect.left + cx - half}px`;
        overlay.style.top = `${rect.top + cy - half}px`;
    };
    return zoom;
}

/**
 * Mirrors Renderer::onMouseEvent/onKeyEvent: passes see canvas mouse, wheel and key events first
 * (e.g. the SplitScreen divider, the SDF editor); one a pass handles stops reaching the camera controller.
 */
function wireMouseForwarding(state: ViewerState): void {
    type PassMouseEvent = { type: "buttonDown" | "buttonUp" | "move" | "wheel"; button?: "left" | "right" | "middle"; pos: [number, number]; wheelDelta?: [number, number] };
    type PassKeyEvent = { type: "keyPressed" | "keyReleased" | "keyRepeated"; key: string; mods: { shift: boolean; ctrl: boolean; alt: boolean } };
    const buttons = ["left", "middle", "right"] as const;
    const dispatch = (ev: Event, call: (p: { onMouseEvent?: (e: PassMouseEvent) => boolean; onKeyEvent?: (e: PassKeyEvent) => boolean }) => boolean | undefined) => {
        let handled = false;
        for (const { pass } of state.graph!.getPasses()) handled = (call(pass as object) ?? false) || handled;
        if (handled) {
            ev.preventDefault();
            ev.stopImmediatePropagation();
        }
    };
    const forward = (ev: MouseEvent, type: PassMouseEvent["type"]) => {
        if (!state.graph) return;
        if ((type === "buttonDown" || type === "wheel") && ev.target !== canvas) return; // the panels' own
        const rect = canvas.getBoundingClientRect();
        const pos: [number, number] = [(ev.clientX - rect.left) / rect.width, (ev.clientY - rect.top) / rect.height];
        // Native wheelDelta.y: +1 = scroll up.
        const wheelDelta: [number, number] | undefined = type === "wheel" ? [0, -Math.sign((ev as WheelEvent).deltaY)] : undefined;
        dispatch(ev, (p) => p.onMouseEvent?.({ type, button: type === "move" || type === "wheel" ? undefined : buttons[ev.button], pos, wheelDelta }));
    };
    // Native Input::Key names from DOM codes ("KeyA" -> "A", "Digit1" -> "Key1", "ShiftLeft" -> "LeftShift").
    const keyName = (code: string) =>
        code.replace(/^Key(?=[A-Z]$)/, "").replace(/^Digit/, "Key").replace(/^(Shift|Control|Alt|Super)(Left|Right)$/, "$2$1").replace(/^Meta(Left|Right)$/, "$1Super");
    const forwardKey = (ev: KeyboardEvent) => {
        if (!state.graph) return;
        const t = ev.target as HTMLElement | null;
        if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
        const type = ev.type === "keyup" ? "keyReleased" : ev.repeat ? "keyRepeated" : "keyPressed";
        const mods = { shift: ev.shiftKey, ctrl: ev.ctrlKey, alt: ev.altKey };
        dispatch(ev, (p) => p.onKeyEvent?.({ type, key: keyName(ev.code), mods }));
    };
    // Capture phase: runs before the camera controller's and the pixel picker's listeners.
    window.addEventListener("mousedown", (ev) => forward(ev, "buttonDown"), true);
    window.addEventListener("mousemove", (ev) => forward(ev, "move"), true);
    window.addEventListener("mouseup", (ev) => forward(ev, "buttonUp"), true);
    window.addEventListener("wheel", (ev) => forward(ev, "wheel"), { capture: true, passive: false });
    window.addEventListener("keydown", forwardKey, true);
    window.addEventListener("keyup", forwardKey, true);
}

/** Render-graph editor panel (Graph button); edits refresh outputs, pass panels and accumulation. */
function wireGraphEditor(state: ViewerState, rebuildUI: () => void, resetAccum: () => void): void {
    const panel = document.getElementById("graphEditor") as HTMLDivElement | null;
    const toggle = document.getElementById("graphToggle") as HTMLButtonElement | null;
    if (!panel || !toggle) return;
    const editor = new GraphEditor(panel, {
        defaultTexDims: () => [canvas.width, canvas.height],
        onGraphChanged: () => {
            state.graphError = null;
            if (state.graph) {
                const outputs = state.graph.getOutputNames();
                if (!state.output || !outputs.includes(state.output)) state.output = outputs[0] ?? null;
            }
            refreshOutputs(state);
            rebuildUI();
            resetAccum();
        },
        onPassPropertiesChanged: () => {
            rebuildUI();
            resetAccum();
        },
    });
    toggle.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        if (!panel.hidden) editor.setGraph(state.graph);
    });
    (window as unknown as { mogwaiGraphEditor: GraphEditor }).mogwaiGraphEditor = editor;
}

/** Wires the plain-DOM control bar (created in index.html). */
function wireControls(state: ViewerState, rebuildUI: () => void): void {
    const $ = (id: string) => document.getElementById(id);
    ($("record") as HTMLButtonElement | null)?.addEventListener("click", (e) => {
        const btn = e.currentTarget as HTMLButtonElement;
        if (!videoRecorder.recording) {
            btn.textContent = "Stop";
            videoRecorder.start(document.getElementById("canvas") as HTMLCanvasElement).catch((err: unknown) => {
                btn.textContent = "Record";
                Logger.error(String(err));
            });
        } else {
            void videoRecorder.stop().then((blob: Blob) => {
                btn.textContent = "Record";
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = "mogwai.webm";
                a.click();
                URL.revokeObjectURL(a.href);
            });
        }
    });
    ($("capture") as HTMLButtonElement | null)?.addEventListener("click", () => {
        void captureFrame(state);
    });
    // Mirrors Mogwai's File > Save Config: the viewer state as a replayable Mogwai script.
    ($("saveConfig") as HTMLButtonElement | null)?.addEventListener("click", () => {
        const script = saveConfig({ graphs: state.graph ? [state.graph] : [], scene: state.scene, scenePath: state.scenePath, width: canvas.width, height: canvas.height, showUI: true, clock: state.clock, frameCapture: state.frameCapture });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([script], { type: "text/x-python" }));
        a.download = "MogwaiConfig.py";
        a.click();
        URL.revokeObjectURL(a.href);
    });
    ($("play") as HTMLButtonElement | null)?.addEventListener("click", () => {
        state.playing = !state.playing;
        ($("play") as HTMLButtonElement).textContent = state.playing ? "Pause" : "Play";
    });
    ($("graphFile") as HTMLInputElement | null)?.addEventListener("change", async (ev) => {
        const file = (ev.target as HTMLInputElement).files?.[0];
        if (file) {
            const [graph] = await runGraphScript(state.device, await file.text(), { frameCapture: state.frameCapture }, state.callbacks);
            await graph!.init();
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
