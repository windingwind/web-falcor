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
    return runMogwaiSource(device, await (await fetch(scriptUrl)).text(), scriptUrl.slice(0, scriptUrl.lastIndexOf("/")), opts);
}

/** Records and replays a Mogwai script's `source`, with local imports resolved against `dirUrl`. */
export async function runMogwaiSource(device: Device, source: string, dirUrl: string, opts: { download?: boolean } = {}): Promise<MogwaiRunResult> {
    const root = "/mogwai";
    const files = await fetchLocalPythonModules(dirUrl, source, root);
    const cwd = `${root}${dirUrl}`;
    const commands: MogwaiCommand[] = recordMogwaiScript(device, source, files, cwd);

    const clock = new Clock();
    const graphs: RenderGraph[] = [];
    let active: RenderGraph | null = null;
    let sceneUpdateCallback: MogwaiCallbacks["sceneUpdateCallback"] = null;
    let scene: Scene | null = null;
    let size: [number, number] = [1920, 1080]; // Mogwai's default frame buffer
    const fc = new FrameCaptureExtension(device, () => active, (name) => graphs.find((g) => g.name === name) ?? null, () => clock.getFrame());
    fc.download = opts.download ?? false;

    const tc = new HeadlessTimingCapture();
    const targetOf = (t: MogwaiTarget): object | null => (t === "clock" ? clock : t === "frameCapture" ? fc : t === "timingCapture" ? tc : scene);
    const resolveRef = (v: unknown): unknown => {
        const ref = v as MogwaiRef | null;
        if (!ref || typeof ref !== "object" || !("mogwaiRef" in ref)) return v;
        const [obj, key] = resolvePath(targetOf(ref.mogwaiRef)!, ref.path);
        const getter = obj[`get${key[0]!.toUpperCase()}${key.slice(1)}`];
        return typeof getter === "function" ? getter.call(obj) : obj[key];
    };

    for (const cmd of commands) {
        switch (cmd.op) {
            case "addGraph":
                graphs.push(cmd.graph);
                // Renderer::addGraph keeps the active graph; the first one added becomes active.
                active ??= cmd.graph;
                cmd.graph.onResize(...size, kTargetFormat);
                if (scene) cmd.graph.setScene(scene);
                await cmd.graph.init();
                break;
            case "removeGraph": {
                // Renderer::removeGraph: the active index steps down past the removed graph.
                const i = graphs.indexOf(cmd.graph);
                let a: number = active ? graphs.indexOf(active) : 0;
                graphs.splice(i, 1);
                if (a >= i && a > 0) a--;
                active = graphs[a] ?? null;
                break;
            }
            case "setActiveGraph":
                active = cmd.graph;
                break;
            case "setSceneUpdateCallback":
                sceneUpdateCallback = cmd.callback;
                break;
            case "loadScene": {
                // os.path.abspath() paths point into the virtual file system: map them back to URLs.
                const path = cmd.path.startsWith(root) ? cmd.path.slice(root.length) : cmd.path;
                const url = path.startsWith("/") ? path : await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
                const baseUrl = url.slice(0, url.lastIndexOf("/"));
                const lower = url.toLowerCase().split(/[?#]/)[0]!;
                const previous = scene;
                const options = { flags: cmd.flags };
                // Mogwai::loadScene: pyscenes run, pbrt parses, everything else goes through the importers.
                scene = lower.endsWith(".pyscene")
                    ? await runSceneScript(device, await (await fetch(url)).text(), baseUrl, { ...options, path: url })
                    : lower.endsWith(".pbrt")
                      ? await runPbrtScene(device, await (await fetch(url)).text(), baseUrl, options)
                      : await runSceneScript(device, `sceneBuilder.importScene(${JSON.stringify(url.slice(baseUrl.length + 1))})`, baseUrl, options);
                if (scene.importPaths[0] !== url) scene.importPaths.unshift(url);
                scene.camera.setAspectRatio(size[0] / size[1]);
                for (const g of graphs) g.setScene(scene);
                previous?.destroy(); // Mogwai frees the replaced scene
                break;
            }
            case "unloadScene":
                // Mirrors Renderer::unloadScene.
                for (const g of graphs) g.setScene(null);
                scene?.destroy();
                scene = null;
                break;
            case "resizeFrameBuffer":
                size = [cmd.width, cmd.height];
                scene?.camera.setAspectRatio(size[0] / size[1]);
                for (const g of graphs) g.onResize(...size, kTargetFormat);
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
                fc.beginFrame();
                tc.beginFrame();
                // Renderer::onFrameRender: the renderer's callback runs before Scene::update.
                if (active) sceneUpdateCallback?.(scene, clock.getTime());
                scene?.runUpdateCallback(clock.getTime());
                if (scene?.isAnimated()) scene.animate(clock.getTime());
                active?.execute(device.renderContext);
                await fc.endFrame();
                break;
        }
    }
    return { frameCapture: fc, timingCapture: tc, graphs, activeGraph: active, scene };
}
