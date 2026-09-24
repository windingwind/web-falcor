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

export interface MogwaiRunResult {
    frameCapture: FrameCaptureExtension;
    graphs: RenderGraph[];
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
    let scene: Scene | null = null;
    let size: [number, number] = [1920, 1080]; // Mogwai's default frame buffer
    const fc = new FrameCaptureExtension(device, () => active, (name) => graphs.find((g) => g.name === name) ?? null, () => clock.getFrame());
    fc.download = opts.download ?? false;

    const targetOf = (t: "clock" | "frameCapture" | "scene"): object | null => (t === "clock" ? clock : t === "frameCapture" ? fc : scene);
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
                active = cmd.graph;
                cmd.graph.onResize(...size, kTargetFormat);
                if (scene) cmd.graph.setScene(scene);
                await cmd.graph.init();
                break;
            case "removeGraph":
                graphs.splice(graphs.indexOf(cmd.graph), 1);
                active = graphs[graphs.length - 1] ?? null;
                break;
            case "loadScene": {
                // os.path.abspath() paths point into the virtual file system: map them back to URLs.
                const path = cmd.path.startsWith(root) ? cmd.path.slice(root.length) : cmd.path;
                const url = path.startsWith("/") ? path : await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
                const baseUrl = url.slice(0, url.lastIndexOf("/"));
                const lower = url.toLowerCase().split(/[?#]/)[0]!;
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
                break;
            }
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
                if (scene?.isAnimated()) scene.animate(clock.getTime());
                active?.execute(device.renderContext);
                await fc.endFrame();
                break;
        }
    }
    return { frameCapture: fc, graphs, scene };
}
