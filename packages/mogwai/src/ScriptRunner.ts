/**
 * Runs an unmodified Mogwai script (e.g. Falcor/tests/image_tests/renderpasses/
 * test_*.py) headlessly: the script is recorded by recordMogwaiScript, then its
 * scene loads, frames and captures are replayed in order through a Clock, the
 * FrameCapture extension and Mogwai's renderFrame sequence.
 */

import {
    AssetCategory,
    AssetResolver,
    Clock,
    ResourceFormat,
    fetchLocalPythonModules,
    recordMogwaiScript,
    runPbrtScene,
    runSceneScript,
    type Device,
    type MogwaiCommand,
    type MogwaiCallbacks,
    type MogwaiTarget,
    type MogwaiRef,
    type RenderGraph,
    type Scene,
} from "@web-falcor/falcor";
import { FrameCaptureExtension } from "./FrameCapture.js";

/** Mogwai sizes graphs from its sRGB swapchain FBO: outputs without a format get this one. */
const kTargetFormat = ResourceFormat.RGBA8UnormSrgb;

/** Walks "a.b.c" to the object holding "c". */
function resolvePath(root: object, path: string): [Record<string, unknown>, string] {
    const parts = path.split(".");
    let obj = root as Record<string, unknown>;
    for (const p of parts.slice(0, -1)) obj = obj[p] as Record<string, unknown>;
    return [obj, parts[parts.length - 1]!];
}

/**
 * Mirrors the TimingCapture extension headlessly: captureFrameTime(path) starts a frame-time
 * log (an empty path stops it); each frame records the previous frame's time in seconds.
 */
export class HeadlessTimingCapture {
    /** Captured logs by path, one frame time (s, Clock::getRealTimeDelta) per entry, as native writes one per line. */
    readonly files = new Map<string, number[]>();
    private current: number[] | null = null;
    private last = -1;
    private frames = 0;

    captureFrameTime(path: string): void {
        this.current = null;
        if (path) this.files.set(String(path), (this.current = []));
    }

    /** TimingCapture::beginFrame: the first valid time is available on the second frame. */
    beginFrame(now = performance.now()): void {
        this.frames++;
        if (this.current && this.frames > 1) this.current.push((now - this.last) / 1000);
        this.last = now;
    }
}

export interface MogwaiRunResult {
    frameCapture: FrameCaptureExtension;
    timingCapture: HeadlessTimingCapture;
    graphs: RenderGraph[];
    /** Renderer's active graph at the end of the script. */
    activeGraph: RenderGraph | null;
    scene: Scene | null;
}

/** Records and replays the Mogwai script at `scriptUrl` (served path, e.g. /Falcor/tests/...). */
export async function runMogwaiScript(device: Device, scriptUrl: string, opts: { download?: boolean } = {}): Promise<MogwaiRunResult> {
    return runMogwaiSource(device, await (await fetch(scriptUrl)).text(), scriptUrl.slice(0, scriptUrl.lastIndexOf("/")), { ...opts, fileName: scriptUrl.slice(scriptUrl.lastIndexOf("/") + 1) });
}

/** Records and replays a Mogwai script's `source`, with local imports resolved against `dirUrl`. */
export async function runMogwaiSource(device: Device, source: string, dirUrl: string, opts: { download?: boolean; fileName?: string } = {}): Promise<MogwaiRunResult> {
    const commands = await recordMogwaiSource(device, source, dirUrl, opts.fileName);
    const clock = new Clock();
    const host: MogwaiHost = {
        graphs: [],
        active: null,
        scene: null,
        size: [1920, 1080], // Mogwai's default frame buffer
        targetFormat: kTargetFormat,
        clock,
        frameCapture: null!,
        timingCapture: new HeadlessTimingCapture(),
        callbacks: { sceneUpdateCallback: null, keyCallback: null },
    };
    host.frameCapture = new FrameCaptureExtension(device, () => host.active, (name) => host.graphs.find((g) => g.name === name) ?? null, () => clock.getFrame());
    host.frameCapture.download = opts.download ?? false;
    await replayMogwaiCommands(device, commands, host);
    return { frameCapture: host.frameCapture, timingCapture: host.timingCapture as HeadlessTimingCapture, graphs: host.graphs, activeGraph: host.active, scene: host.scene };
}

/** The script's local modules are fetched and it is recorded (see recordMogwaiScript). */
export async function recordMogwaiSource(device: Device, source: string, dirUrl: string, fileName?: string): Promise<MogwaiCommand[]> {
    const files = await fetchLocalPythonModules(dirUrl, source, kScriptRoot);
    return recordMogwaiScript(device, source, files, `${kScriptRoot}${dirUrl}`, fileName);
}

/** Pyodide directory recorded scripts run in (os.path.abspath() paths start with it). */
const kScriptRoot = "/mogwai";

/** The Renderer state a recorded script acts on: the headless runner's own, or the viewer's. */
export interface MogwaiHost {
    graphs: RenderGraph[];
    active: RenderGraph | null;
    scene: Scene | null;
    /** Frame buffer size (m.resizeFrameBuffer). */
    size: [number, number];
    /** Format given to graphs without an output format (the swapchain's); undefined = the graph default. */
    targetFormat?: ResourceFormat;
    clock: Clock;
    frameCapture: FrameCaptureExtension;
    timingCapture: { beginFrame(): void; captureFrameTime(path: string): void };
    callbacks: MogwaiCallbacks;
    /** Mogwai::loadScene for a resolved URL; the default runs pyscenes, parses pbrt and imports the rest. */
    loadScene?(url: string, baseUrl: string, flags: number): Promise<Scene>;
    /** Called after m.resizeFrameBuffer. */
    onResize?(width: number, height: number): void;
}

/** Mogwai::loadScene: pyscenes run, pbrt parses, everything else goes through the importers. */
async function loadSceneDefault(device: Device, url: string, baseUrl: string, flags: number): Promise<Scene> {
    const lower = url.toLowerCase().split(/[?#]/)[0]!;
    const options = { flags };
    return lower.endsWith(".pyscene")
        ? runSceneScript(device, await (await fetch(url)).text(), baseUrl, { ...options, path: url })
        : lower.endsWith(".pbrt")
          ? runPbrtScene(device, await (await fetch(url)).text(), baseUrl, options)
          : runSceneScript(device, `sceneBuilder.importScene(${JSON.stringify(url.slice(baseUrl.length + 1))})`, baseUrl, options);
}

/** Replays recorded commands in order: scene loads, graph edits, frames and captures (Mogwai's renderFrame sequence). */
export async function replayMogwaiCommands(device: Device, commands: MogwaiCommand[], host: MogwaiHost): Promise<void> {
    const { clock } = host;
    const targetOf = (t: MogwaiTarget): object | null => (t === "clock" ? clock : t === "frameCapture" ? host.frameCapture : t === "timingCapture" ? host.timingCapture : host.scene);
    const resolveRef = (v: unknown): unknown => {
        const ref = v as MogwaiRef | null;
        if (!ref || typeof ref !== "object" || !("mogwaiRef" in ref)) return v;
        const [obj, key] = resolvePath(targetOf(ref.mogwaiRef)!, ref.path);
        const getter = obj[`get${key[0]!.toUpperCase()}${key.slice(1)}`];
        return typeof getter === "function" ? getter.call(obj) : obj[key];
    };
    const { graphs } = host;

    for (const cmd of commands) {
        switch (cmd.op) {
            case "addGraph": {
                // Renderer::addGraph: a graph of the same name is replaced in place; the active graph
                // stays (the first one added becomes active).
                const same = graphs.findIndex((g) => g.name === cmd.graph.name);
                if (same >= 0) {
                    if (host.active === graphs[same]) host.active = cmd.graph;
                    graphs[same] = cmd.graph;
                } else graphs.push(cmd.graph);
                host.active ??= cmd.graph;
                cmd.graph.onResize(...host.size, host.targetFormat);
                if (host.scene) cmd.graph.setScene(host.scene);
                await cmd.graph.init();
                break;
            }
            case "removeGraph": {
                // Renderer::removeGraph: the active index steps down past the removed graph.
                const i = graphs.indexOf(cmd.graph);
                let a: number = host.active ? graphs.indexOf(host.active) : 0;
                graphs.splice(i, 1);
                if (a >= i && a > 0) a--;
                host.active = graphs[a] ?? null;
                break;
            }
            case "setActiveGraph":
                host.active = cmd.graph;
                break;
            case "setSceneUpdateCallback":
                host.callbacks.sceneUpdateCallback = cmd.callback;
                break;
            case "setKeyCallback":
                host.callbacks.keyCallback = cmd.callback;
                break;
            case "loadScene": {
                // os.path.abspath() paths point into the virtual file system: map them back to URLs.
                const path = cmd.path.startsWith(kScriptRoot) ? cmd.path.slice(kScriptRoot.length) : cmd.path;
                const url = path.startsWith("/") ? path : await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
                const baseUrl = url.slice(0, url.lastIndexOf("/"));
                const previous = host.scene;
                const scene = await (host.loadScene ? host.loadScene(url, baseUrl, cmd.flags) : loadSceneDefault(device, url, baseUrl, cmd.flags));
                if (scene.importPaths[0] !== url) scene.importPaths.unshift(url);
                scene.camera.setAspectRatio(host.size[0] / host.size[1]);
                host.scene = scene;
                for (const g of graphs) g.setScene(scene);
                previous?.destroy(); // Mogwai frees the replaced scene
                break;
            }
            case "unloadScene":
                // Mirrors Renderer::unloadScene.
                for (const g of graphs) g.setScene(null);
                host.scene?.destroy();
                host.scene = null;
                break;
            case "resizeFrameBuffer":
                host.size = [cmd.width, cmd.height];
                host.scene?.camera.setAspectRatio(cmd.width / cmd.height);
                for (const g of graphs) g.onResize(cmd.width, cmd.height, host.targetFormat);
                host.onResize?.(cmd.width, cmd.height);
                break;
            case "set": {
                const target = targetOf(cmd.target);
                if (!target) throw new Error(`m.${cmd.target}.${cmd.key} set before a scene is loaded`);
                const [obj, key] = resolvePath(target, cmd.key);
                const value = resolveRef(cmd.value);
                // Python properties map to setX() where the web object has one (it marks state dirty).
                const setter = obj[`set${key[0]!.toUpperCase()}${key.slice(1)}`];
                if (typeof setter === "function") setter.call(obj, value);
                else obj[key] = value;
                break;
            }
            case "call": {
                if (typeof cmd.target === "string") {
                    const target = targetOf(cmd.target);
                    if (!target) throw new Error(`m.${cmd.target}.${cmd.method}() before a scene is loaded`);
                    const [obj, key] = resolvePath(target, cmd.method);
                    await (obj[key] as (...a: unknown[]) => unknown).apply(obj, cmd.args.map(resolveRef));
                } else {
                    (cmd.target as unknown as Record<string, (...a: unknown[]) => unknown>)[cmd.method]!.apply(cmd.target, cmd.args);
                    for (const g of graphs) await g.init();
                }
                break;
            }
            case "setPass":
                (cmd.target as unknown as Record<string, unknown>)[cmd.key] = cmd.value;
                break;
            case "renderFrame":
                // Mogwai::renderFrame: clock, extensions' beginFrame, scene update, graph, endFrame.
                clock.tick();
                host.frameCapture.beginFrame();
                host.timingCapture.beginFrame();
                // Renderer::onFrameRender: the renderer's callback runs before Scene::update.
                if (host.active) host.callbacks.sceneUpdateCallback?.(host.scene, clock.getTime());
                host.scene?.runUpdateCallback(clock.getTime());
                if (host.scene?.isAnimated()) host.scene.animate(clock.getTime());
                host.active?.execute(device.renderContext);
                await host.frameCapture.endFrame();
                break;
        }
    }
}
