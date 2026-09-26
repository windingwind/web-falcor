/**
 * Mogwai (web) — the interactive viewer: load a render-graph .py + .pyscene,
 * execute the graph each frame, present the marked output to the canvas.
 */

import { FrameCaptureExtension, captureOutput } from "./FrameCapture.js";
import { recordMogwaiSource, replayMogwaiCommands, type MogwaiHost } from "./ScriptRunner.js";
import { AssetCategory, AssetResolver, isAbsoluteUrl, kProjectMediaUrl, Clock, Device, Logger, LogLevel, SceneBuilderFlags, getGlobalSettings, Profiler, ProfilerUI, VideoRecorder, ProgramManager, RenderGraph, ResourceFormat, Bitmap, BitmapExportFlags, createPass, initScripting, initSlang, runConsoleCommand, runSceneScript, DomWidgets, nativeKeyCode, GamepadInput, GamepadEventType, GamepadButton, type MogwaiCallbacks, runPbrtScene, runMitsubaScene, presentToCanvas, OverlayDrawList, type Scene } from "@web-falcor/falcor";
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
    /** Every graph the script added (Renderer::mGraphs); `graph` is the active one. */
    graphs: RenderGraph[];
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
    await runScriptSource(state, await (await fetch(url)).text(), url.slice(0, url.lastIndexOf("/")), url.split(/[?#]/)[0]!.slice(url.lastIndexOf("/") + 1));
}

/** The viewer's built-in graph (native starts without one): a script's graphs replace it. */
let builtInGraph: RenderGraph | null = null;

/**
 * Renderer::loadScript: the script runs with the full `m` (addGraph, loadScene, m.scene edits,
 * resizeFrameBuffer, renderFrame, captures, callbacks), recorded and replayed onto the viewer's state.
 */
async function runScriptSource(state: ViewerState, source: string, dirUrl: string, fileName?: string): Promise<void> {
    const commands = await recordMogwaiSource(state.device, source, dirUrl, fileName);
    if (builtInGraph && state.graphs.includes(builtInGraph)) {
        state.graphs.splice(state.graphs.indexOf(builtInGraph), 1);
        if (state.graph === builtInGraph) state.graph = null;
        builtInGraph = null;
    }
    const host: MogwaiHost = {
        graphs: state.graphs,
        get active() { return state.graph; },
        set active(g) { state.graph = g; },
        get scene() { return state.scene; },
        set scene(scene) {
            state.scene = scene;
            state.scenePath = scene?.importPaths[0] ?? null;
        },
        get size(): [number, number] { return [canvas.width, canvas.height]; },
        set size([w, h]) { [canvas.width, canvas.height] = [w, h]; },
        clock: state.clock,
        frameCapture: state.frameCapture!,
        timingCapture: { beginFrame: () => {}, captureFrameTime: (path) => void state.timingCapture.captureFrameTime(path) },
        callbacks: state.callbacks,
        loadScene: (url, baseUrl, flags) => createScene(state, url, baseUrl, flags),
    };
    await replayMogwaiCommands(state.device, commands, host);
    selectGraph(state, state.graph ?? state.graphs[0] ?? null);
    // The script's callbacks run later and look `m` up then: rebind it to the live viewer (the console's m).
    runConsoleCommand(state.device, "", liveConsoleContext(state));
}

/** The console's `m` over the viewer's live state. */
function liveConsoleContext(state: ViewerState): Parameters<typeof runConsoleCommand>[2] {
    return {
        get scene() { return state.scene; },
        get graph() { return state.graph; },
        clock: state.clock,
        timingCapture: state.timingCapture,
        frameCapture: state.frameCapture,
        callbacks: state.callbacks,
    };
}

/** Resizes the frame buffer (m.resizeFrameBuffer / the Window Size setting): canvas, graphs and camera aspect. */
function resizeFrameBuffer(state: ViewerState, width: number, height: number): void {
    [canvas.width, canvas.height] = [width, height];
    for (const g of state.graphs) g.onResize(width, height);
    state.scene?.camera.setAspectRatio(width / height);
    state.frame = 0;
}

/** MogwaiSettings::selectNextGraph / the graph dropdown: switches the active graph. */
function selectGraph(state: ViewerState, graph: RenderGraph | null): void {
    state.graph = graph;
    graph?.onResize(canvas.width, canvas.height);
    state.output = graph?.getOutputNames()[0] ?? null;
    state.frame = 0;
}

/** Builds the scene at `url` (Mogwai::loadScene) without installing it. */
async function createScene(state: ViewerState, url: string, baseUrl: string, flags?: number): Promise<Scene> {
    const lower = url.toLowerCase().split(/[?#]/)[0]!;
    if (rebuildSceneCache) flags = (flags ?? SceneBuilderFlags.Default) | SceneBuilderFlags.RebuildCache;
    rebuildSceneCache = false;
    const scene = lower.endsWith(".pbrt")
        ? await runPbrtScene(state.device, await (await fetch(url)).text(), baseUrl, { flags })
        : lower.endsWith(".xml") // Mitsuba scenes are the only .xml we load
          ? await runMitsubaScene(state.device, await (await fetch(url)).text(), baseUrl)
          : lower.endsWith(".pyscene")
            ? await runSceneScript(state.device, await (await fetch(url)).text(), baseUrl, { cache: true, path: url, flags }) // OPFS scene cache: fast reloads
            : await runSceneScript(state.device, `sceneBuilder.importScene(${JSON.stringify(url.slice(baseUrl.length + 1))})`, baseUrl, { cache: true, path: url, flags });
    if (scene.importPaths[0] !== url) scene.importPaths.unshift(url);
    return scene;
}

async function loadScene(state: ViewerState, url: string, baseUrl: string): Promise<void> {
    installScene(state, await createScene(state, url, baseUrl), url);
}

/** Makes `scene` the viewer's scene (Renderer::setScene), freeing the previous one. */
function installScene(state: ViewerState, scene: Scene, url: string | null): void {
    scene.camera.setAspectRatio(canvas.width / canvas.height);
    const previous = state.scene;
    state.scene = scene;
    state.scenePath = url;
    for (const g of state.graphs) g.setScene(scene);
    state.frame = 0;
    previous?.destroy(); // Mogwai frees the replaced scene
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
        addRecent("scenes", sceneParam);
    }
    if (graphParam) {
        await loadGraph(state, await resolveAssetUrl(graphParam));
        addRecent("scripts", graphParam);
    } else if (!sceneParam) {
        // No URL content: default cornell box + the GPU-oracle-verified graph.
        await loadScene(state, "/Falcor/media/test_scenes/cornell_box.pyscene", "/Falcor/media/test_scenes");
        state.graph = builtInGraph = buildDefaultGraph(device, canvas.width, canvas.height, state.scene!);
        state.graphs = [state.graph];
        state.output = state.graph.getOutputNames()[0] ?? null;
    } else {
        // Scene but no graph: run the default path tracer over the chosen scene.
        state.graph = builtInGraph = buildDefaultGraph(device, canvas.width, canvas.height, state.scene!);
        state.graphs = [state.graph];
        state.output = state.graph.getOutputNames()[0] ?? null;
    }
    if (outputParam && state.graph?.getOutputNames().includes(outputParam)) {
        state.output = outputParam;
    }
}

const videoRecorder = new VideoRecorder();

/**
 * Mogwai's command-line options as URL parameters: `verbosity` (0-5), `width`/`height`, `attributes`
 * (URL of a JSON attributes file), `rebuild-cache`, `headless` (canvas only), `fullscreen` (entered on
 * the first click: browsers need a gesture) and `deferred` (content loads on the next animation frame).
 */
async function applyCommandLineParams(): Promise<{ deferred: boolean }> {
    const params = new URLSearchParams(location.search);
    const verbosity = params.get("verbosity") ?? params.get("v");
    if (verbosity !== null) {
        const level = Number(verbosity);
        if (!Number.isInteger(level) || level < LogLevel.Disabled || level > LogLevel.Debug) throw new Error(`Mogwai: invalid verbosity level ${verbosity}`);
        Logger.level = level;
    }
    const [width, height] = [Number(params.get("width")), Number(params.get("height"))];
    if (width > 0) canvas.width = width;
    if (height > 0) canvas.height = height;
    const attributes = params.get("attributes") ?? params.get("a");
    if (attributes) {
        const res = await fetch(await resolveAssetUrl(attributes));
        if (!res.ok) throw new Error(`Failed to load attributes file '${attributes}'.`);
        getGlobalSettings().addFilteredAttributes(await res.json());
    }
    rebuildSceneCache = params.has("rebuild-cache");
    if (params.has("headless") || params.has("silent")) {
        document.body.classList.add("headless");
        recordRecent = false;
    }
    if (params.has("fullscreen")) window.addEventListener("pointerdown", () => void canvas.requestFullscreen().catch(() => {}), { once: true });
    return { deferred: params.has("deferred") };
}
/** `?rebuild-cache`: the first scene load rebuilds its cache entry (SceneBuilder::Flags::RebuildCache). */
let rebuildSceneCache = false;

// Mogwai AppData's recent scripts/scenes (25 each, most recent first), kept in localStorage.
// Silent/headless runs don't record, as natively (image tests use them).
type RecentKind = "scripts" | "scenes";
let recordRecent = true;
function getRecent(kind: RecentKind): string[] {
    try {
        const list = JSON.parse(localStorage.getItem(`mogwai.recent.${kind}`) ?? "[]") as unknown;
        return Array.isArray(list) ? list.filter((p): p is string => typeof p === "string") : [];
    } catch {
        return [];
    }
}
function addRecent(kind: RecentKind, path: string): void {
    if (!recordRecent) return;
    localStorage.setItem(`mogwai.recent.${kind}`, JSON.stringify([path, ...getRecent(kind).filter((p) => p !== path)].slice(0, 25)));
    refreshRecentList();
}
function refreshRecentList(): void {
    const list = document.getElementById("recentList");
    if (!list) return;
    list.replaceChildren(
        ...[...getRecent("scenes").map((p) => [p, "recent scene"]), ...getRecent("scripts").map((p) => [p, "recent script"])].map(([value, label]) => {
            const o = document.createElement("option");
            o.value = value!;
            o.label = label!;
            return o;
        }),
    );
}

async function main() {
    // Before the device exists, so verbosity also covers device creation (as the native flags do).
    const { deferred } = await applyCommandLineParams();
    const device = await Device.create();
    const profiler = new Profiler(device);
    if (profiler.available) device.enableProfiler(profiler);
    const context = canvas.getContext("webgpu");
    if (!context) throw new Error("Failed to get webgpu canvas context");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device: device.gpuDevice, format });

    await initProgramSystem(device);
    await initScripting("/node_modules/pyodide");

    const state: ViewerState = { device, context, format, graph: null, graphs: [], scene: null, output: null, frame: 0, playing: true, clock: new Clock(), timingCapture: new TimingCapture(), frameCapture: null, animateScene: true, graphError: null, scenePath: null, callbacks: { sceneUpdateCallback: null, keyCallback: null } };
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
    state.frameCapture = new FrameCaptureExtension(device, () => state.graph, (name) => (state.graph?.name === name ? state.graph : null), () => state.clock.getFrame());

    // Initial content from URL params (?scene=/?graph=/?output=), or the default
    // cornell-box path tracer when none are given.
    if (deferred) await new Promise((r) => requestAnimationFrame(r)); // Renderer::loadScriptDeferred
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
    const camControl = new CameraController(canvas, () => state.scene);
    const gamepad = new GamepadInput();
    /** MogwaiSettings::selectNextGraph (N key, gamepad Y). */
    const selectNextGraph = () => {
        if (state.graphs.length < 2) return;
        selectGraph(state, state.graphs[(state.graphs.indexOf(state.graph!) + 1) % state.graphs.length]!);
        refreshOutputs(state);
        rebuildUI();
    };
    camControl.onViewpointAdded = () => rebuildUI();
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
                setSpeed: (v) => camControl.setSpeed(v),
            },
        }, state.device.programManager);

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
    // MogwaiSettings::renderTimeSettings (F9): the clock's UI, refreshed while shown (not mid-interaction).
    const timePanel = document.getElementById("time") as HTMLDivElement;
    const renderTimePanel = () => {
        if (timePanel.hidden) return;
        timePanel.replaceChildren();
        const ui = new DomWidgets(timePanel, renderTimePanel);
        ui.text("Time");
        state.clock.renderUI(ui);
        const [exitTime, exitFrame] = [state.clock.getExitTime(), state.clock.getExitFrame()];
        if (exitTime || exitFrame) ui.text(`Exiting in ${exitTime ? `${(exitTime - state.clock.getTime()).toFixed(2)} seconds` : ""}${exitFrame ? `${exitFrame - state.clock.getFrame()} frames` : ""}`);
    };
    setInterval(() => {
        if (!timePanel.hidden && !timePanel.matches(":hover") && !timePanel.contains(document.activeElement)) renderTimePanel();
    }, 250);
    window.addEventListener("mogwai-graphchange", () => rebuildUI());
    // MogwaiSettings::keyboardEvent: F1 help, F7 overlay UI, F9 time panel, F10 the FPS line, N the next graph.
    const settingsKey = (ev: KeyboardEvent): boolean => {
        if (ev.ctrlKey || ev.shiftKey || ev.altKey) return false;
        if (ev.key === "F1") {
            const help = document.getElementById("help");
            if (help) help.hidden = !help.hidden;
        } else if (ev.key === "F7") overlayCanvas.hidden = !overlayCanvas.hidden;
        else if (ev.key === "F9") {
            timePanel.hidden = !timePanel.hidden;
            renderTimePanel();
        } else if (ev.key === "F10") status.hidden = !status.hidden;
        else if (ev.key === "n" || ev.key === "N") selectNextGraph();
        else return false;
        ev.preventDefault();
        return true;
    };
    // SampleApp::handleKeyboardEvent: Space/Pause the clock, Ctrl+Space the renderer, F2 all UI,
    // F12 screen capture, P the profiler, F6 shader reload (native F5 is the browser's, docs §9), ` the console.
    const appKey = (ev: KeyboardEvent): void => {
        const modified = ev.ctrlKey || ev.shiftKey || ev.altKey;
        if ((ev.key === " " || ev.key === "Pause") && !ev.shiftKey && !ev.altKey) {
            ev.preventDefault();
            if (ev.ctrlKey) (document.getElementById("play") as HTMLButtonElement | null)?.click();
            else if (state.clock.isPaused()) state.clock.play();
            else state.clock.pause();
        } else if (modified) {
            return;
        } else if (ev.key === "p" || ev.key === "P") {
            profilerPanel.hidden = !profilerPanel.hidden;
        } else if (ev.key === "F2") {
            ev.preventDefault();
            document.body.classList.toggle("headless");
        } else if (ev.key === "F6") {
            ev.preventDefault();
            void reloadShaders(state.device).then(() => (state.frame = 0));
        } else if (ev.key === "F12") {
            ev.preventDefault();
            (document.getElementById("capture") as HTMLButtonElement | null)?.click();
        } else if (ev.key === "`") {
            ev.preventDefault();
            (document.getElementById("consoleToggle") as HTMLButtonElement | null)?.click();
        }
    };
    // Renderer::onKeyEvent then SampleApp: MogwaiSettings, the scene (camera, F3), m.keyCallback, the app's hotkeys
    // (the graph's passes already saw the event in the capture phase).
    const onViewerKey = (ev: KeyboardEvent) => {
        if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return;
        const pressed = ev.type === "keydown";
        if (pressed && settingsKey(ev)) return;
        if (camControl.handleKey(ev)) return;
        const cb = state.callbacks.keyCallback;
        if (cb && !ev.repeat && runRendererCallback("keyCallback", () => cb(pressed, nativeKeyCode(ev.code)))) {
            ev.preventDefault();
            return;
        }
        if (pressed) appKey(ev);
    };
    window.addEventListener("keydown", onViewerKey);
    window.addEventListener("keyup", onViewerKey);
    (window as unknown as { mogwaiProfiler: { profiler: Profiler; ui: ProfilerUI } }).mogwaiProfiler = { profiler, ui: profilerUI };
    const pixelZoom = wirePixelZoom();
    (window as unknown as { mogwaiPixelZoom: PixelZoom }).mogwaiPixelZoom = pixelZoom;
    (window as unknown as { mogwai: ViewerState }).mogwai = state; // debug/test handle
    (window as unknown as { mogwaiCamControl: CameraController }).mogwaiCamControl = camControl;

    let lastGpuLine = "";
    let lastNow = -1;
    function frame(now: number) {
        // Renderer::onFrameRender: m.sceneUpdateCallback, then Scene::update (the scene's python updateCallback first).
        if (state.graph) runRendererCallback("sceneUpdateCallback", () => state.callbacks.sceneUpdateCallback?.(state.scene, state.clock.getTime()));
        state.scene?.runUpdateCallback(state.clock.getTime());
        // Window::pollForEvents: MogwaiSettings takes Y (next graph), the scene's camera the sticks.
        gamepad.poll({
            handleGamepadEvent: (e) => {
                if (e.type === GamepadEventType.ButtonDown && e.button === GamepadButton.Y) selectNextGraph();
            },
            handleGamepadState: (s) => camControl.onGamepadState(s),
        });
        let dirty = state.scene ? camControl.update(state.scene, now) : false;
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
            presentDebugWindows(state);
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
    // MogwaiSettings::winSizeUI: the renderable area (swapchain) size, common resolutions or custom.
    const sizeSel = $("winSize") as HTMLSelectElement | null;
    if (sizeSel) {
        const kResolutions = ["1280x720", "1920x1080", "1920x1200", "2560x1440", "3840x2160"];
        const sync = () => {
            const cur = `${canvas.width}x${canvas.height}`;
            sizeSel.replaceChildren(...[...kResolutions, ...(kResolutions.includes(cur) ? [] : [cur]), "Custom…"].map((r) => Object.assign(document.createElement("option"), { value: r, textContent: r })));
            sizeSel.value = cur;
        };
        sync();
        sizeSel.onchange = () => {
            const v = sizeSel.value === "Custom…" ? prompt("Window size (WIDTHxHEIGHT)", `${canvas.width}x${canvas.height}`) : sizeSel.value;
            const m = v?.match(/^\s*(\d+)\s*[xX*]\s*(\d+)\s*$/);
            if (m && +m[1]! > 0 && +m[2]! > 0) resizeFrameBuffer(state, +m[1]!, +m[2]!);
            sync();
        };
    }
    // Renderer::onDroppedFile: a dropped script runs, a dropped scene loads. A dropped file has no
    // directory: pyscenes resolve assets against the media root, binary formats must be self-contained.
    window.addEventListener("dragover", (ev) => ev.preventDefault());
    window.addEventListener("drop", (ev) => {
        const file = ev.dataTransfer?.files[0];
        if (!file) return;
        ev.preventDefault();
        const ext = file.name.slice(file.name.lastIndexOf(".") + 1).toLowerCase();
        void (async () => {
            if (ext === "py") await runScriptSource(state, await file.text(), location.pathname.replace(/\/[^/]*$/, ""), file.name);
            else if (ext === "pyscene") installScene(state, await runSceneScript(state.device, await file.text(), kProjectMediaUrl, { path: file.name }), null);
            else if (ext === "pbrt") installScene(state, await runPbrtScene(state.device, await file.text(), kProjectMediaUrl), null);
            else if (["fbx", "gltf", "glb", "obj", "usd", "usda", "usdc", "usdz", "dae", "3ds", "ply", "blend"].includes(ext)) {
                // The fragment carries the extension the importer dispatches on (fetching a blob URL ignores it).
                const url = `${URL.createObjectURL(file)}#${file.name}`;
                installScene(state, await runSceneScript(state.device, `sceneBuilder.importScene(${JSON.stringify(url)})`, kProjectMediaUrl), null);
            } else return Logger.warning(`RenderGraphViewer::onDroppedFile() - Unknown file extension '${ext}'`);
            refreshOutputs(state);
            rebuildUI();
        })().catch((err: unknown) => Logger.error(`Mogwai: failed to load dropped '${file.name}': ${String(err)}`));
    });
    // File > Load Script / Load Scene and the recent lists (Enter loads the typed or picked path).
    const open = $("openPath") as HTMLInputElement | null;
    refreshRecentList();
    open?.addEventListener("keydown", (ev) => {
        if (ev.key !== "Enter" || !open.value.trim()) return;
        const path = open.value.trim();
        const isScript = /\.py$/i.test(path.split(/[?#]/)[0]!);
        status.textContent = `loading ${path}…`;
        void (async () => {
            const url = await resolveAssetUrl(path);
            if (isScript) await loadGraph(state, url);
            else await loadScene(state, url, url.slice(0, url.lastIndexOf("/")));
            addRecent(isScript ? "scripts" : "scenes", path);
            refreshOutputs(state);
            rebuildUI();
            open.value = "";
        })().catch((err: unknown) => Logger.error(`Mogwai: failed to load '${path}': ${String(err)}`));
    });
    window.addEventListener("keydown", (ev) => {
        if (ev.ctrlKey && ev.key.toLowerCase() === "o" && open) {
            ev.preventDefault(); // the browser's own file dialog
            open.placeholder = ev.shiftKey ? "scene path" : "script path (.py)";
            open.focus();
        }
    });
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
        const script = saveConfig({ graphs: state.graphs, scene: state.scene, scenePath: state.scenePath, width: canvas.width, height: canvas.height, showUI: true, clock: state.clock, frameCapture: state.frameCapture });
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
            await runScriptSource(state, await file.text(), location.pathname.replace(/\/[^/]*$/, ""), file.name);
            refreshOutputs(state);
            rebuildUI();
        }
    });
    refreshOutputs(state);
}

function refreshOutputs(state: ViewerState): void {
    // The active-graph dropdown (shown for scripts with several graphs; N cycles them).
    const graphSel = document.getElementById("activeGraph") as HTMLSelectElement | null;
    if (graphSel) {
        graphSel.replaceChildren(...state.graphs.map((g, i) => Object.assign(document.createElement("option"), { value: String(i), textContent: g.name })));
        graphSel.value = String(Math.max(0, state.graphs.indexOf(state.graph!)));
        graphSel.parentElement!.hidden = state.graphs.length < 2;
        graphSel.onchange = () => {
            selectGraph(state, state.graphs[Number(graphSel.value)] ?? null);
            refreshOutputs(state);
            window.dispatchEvent(new Event("mogwai-graphchange"));
        };
        // Renderer::removeGraph: the active index steps down past the removed graph.
        (document.getElementById("removeGraph") as HTMLButtonElement).onclick = (ev) => {
            ev.preventDefault();
            const i = state.graphs.indexOf(state.graph!);
            if (i < 0) return;
            state.graphs.splice(i, 1);
            selectGraph(state, state.graphs[i > 0 ? i - 1 : 0] ?? null);
            refreshOutputs(state);
            window.dispatchEvent(new Event("mogwai-graphchange"));
        };
    }
    const sel = document.getElementById("output") as HTMLSelectElement | null;
    if (!sel || !state.graph) return;
    const graph = state.graph;
    const data = graphDataOf(graph);
    // Renderer::graphOutputsGui: debug windows force the full list.
    const names = outputChoices(graph);
    // renderOutputUI: an output that left the list (List All unchecked) is replaced by the first one.
    if (!state.output || !names.includes(state.output)) setMainOutput(state, names[0] ?? null);
    sel.replaceChildren(...names.map((name) => Object.assign(document.createElement("option"), { value: name, textContent: name })));
    if (state.output) sel.value = state.output;
    sel.onchange = () => setMainOutput(state, sel.value);
    const all = document.getElementById("allOutputs") as HTMLInputElement | null;
    if (all) {
        all.checked = data.showAll || debugWindows.length > 0;
        all.disabled = debugWindows.length > 0;
        all.onchange = () => {
            data.showAll = all.checked;
            refreshOutputs(state);
        };
    }
    const dbg = document.getElementById("debugWindow") as HTMLButtonElement | null;
    if (dbg) dbg.onclick = () => addDebugWindow(state);
}

/** Renderer::GraphData: the graph's own outputs, and reference counts for outputs marked only for display. */
interface GraphData {
    originalOutputs: string[];
    refs: Map<string, number>;
    showAll: boolean;
}
const graphData = new WeakMap<RenderGraph, GraphData>();
function graphDataOf(graph: RenderGraph): GraphData {
    let data = graphData.get(graph);
    if (!data) graphData.set(graph, (data = { originalOutputs: graph.getOutputNames(), refs: new Map(), showAll: false }));
    return data;
}
function outputChoices(graph: RenderGraph): string[] {
    const data = graphDataOf(graph);
    return data.showAll || debugWindows.length > 0 ? graph.getAvailableOutputs() : data.originalOutputs;
}
/** Renderer::markOutput / unmarkOutput: original outputs stay; others are marked while something shows them. */
function markViewerOutput(graph: RenderGraph, name: string): void {
    const data = graphDataOf(graph);
    if (data.originalOutputs.includes(name)) return;
    const n = (data.refs.get(name) ?? 0) + 1;
    data.refs.set(name, n);
    if (n === 1) graph.markOutput(name);
}
function unmarkViewerOutput(graph: RenderGraph, name: string): void {
    const data = graphDataOf(graph);
    if (data.originalOutputs.includes(name) || !data.refs.has(name)) return;
    const n = data.refs.get(name)! - 1;
    if (n > 0) return void data.refs.set(name, n);
    data.refs.delete(name);
    graph.unmarkOutput(name);
}
function setMainOutput(state: ViewerState, name: string | null): void {
    const graph = state.graph;
    if (!graph) return;
    if (state.output) unmarkViewerOutput(graph, state.output);
    state.output = name;
    if (name) markViewerOutput(graph, name);
}

/** Renderer's debug windows ("Show In Debug Window"): another output of the active graph, presented each frame. */
interface DebugWindow {
    graph: RenderGraph;
    output: string;
    el: HTMLDivElement;
    ctx: GPUCanvasContext;
}
const debugWindows: DebugWindow[] = [];
let debugWindowIndex = 0;
function addDebugWindow(state: ViewerState): void {
    const graph = state.graph;
    if (!graph || !state.output) return;
    const el = document.createElement("div");
    el.className = "debug-window";
    const title = document.createElement("div");
    title.className = "debug-title";
    title.textContent = `Debug Window ${debugWindowIndex++}`;
    const close = Object.assign(document.createElement("button"), { textContent: "×", title: "Close" });
    const sel = document.createElement("select");
    const save = Object.assign(document.createElement("button"), { textContent: "Save To File" });
    const canvasEl = document.createElement("canvas");
    [canvasEl.width, canvasEl.height] = [Math.round(canvas.width * 0.4), Math.round(canvas.height * 0.55)];
    title.append(close);
    el.append(title, sel, save, canvasEl);
    document.body.appendChild(el);
    const ctx = canvasEl.getContext("webgpu")!;
    ctx.configure({ device: state.device.gpuDevice, format: state.format });
    const win: DebugWindow = { graph, output: state.output, el, ctx };
    markViewerOutput(graph, win.output);
    debugWindows.push(win);
    const fill = () => {
        sel.replaceChildren(...outputChoices(graph).map((n) => Object.assign(document.createElement("option"), { value: n, textContent: n })));
        sel.value = win.output;
    };
    fill();
    sel.onchange = () => {
        unmarkViewerOutput(graph, win.output);
        win.output = sel.value;
        markViewerOutput(graph, win.output);
    };
    save.onclick = () => void graph.getOutput(win.output)?.captureToFile(0, 0, `${win.output.replace(/\./g, "_")}.png`, Bitmap.getFormatFromFileExtension("png"));
    close.onclick = () => {
        unmarkViewerOutput(graph, win.output);
        debugWindows.splice(debugWindows.indexOf(win), 1);
        el.remove();
        refreshOutputs(state);
    };
    refreshOutputs(state);
}
function presentDebugWindows(state: ViewerState): void {
    for (const w of debugWindows) {
        if (w.graph !== state.graph) continue;
        const tex = w.graph.getOutput(w.output);
        if (tex) presentToCanvas(state.device, tex, w.ctx.getCurrentTexture(), state.format);
    }
}

main().catch((err) => {
    Logger.error(String(err));
    status.textContent = `FAILED: ${err.message ?? err}`;
});
