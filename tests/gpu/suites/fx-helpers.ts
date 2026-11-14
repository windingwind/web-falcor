/**
 * Shared fixtures of the Hi-Z, history, DepthComposite and ParticleLayer tests: graphs of any size with external inputs,
 * camera poses, and the cold-vs-consecutive protocol for passes that keep state across frames.
 */

import { RenderGraph, ResourceBindFlags, ResourceFormat, createPass, float3, type Device, type Scene, type Texture } from "@web-falcor/falcor";
import { resetGraphHistory } from "@web-falcor/render-passes";
import { bitwiseEqual, read, type PassDecl, type V3 } from "./stylize-helpers.js";

let graphCount = 0;

/** GBufferRaster "G" + passes; no execution. */
export function buildGraph(device: Device, scene: Scene, passes: PassDecl[], edges: [string, string][], outputs: string[],
    w: number, h: number, inputs: Record<string, Texture> = {}): RenderGraph {
    const g = new RenderGraph(device, `FxTest${++graphCount}`);
    g.onResize(w, h);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    for (const [type, id, props] of passes) g.addPass(createPass(device, type, props ?? {}), id);
    for (const [a, b] of edges) g.addEdge(a, b);
    for (const [ref, t] of Object.entries(inputs)) g.setInput(ref, t);
    for (const o of new Set([...outputs, "G.posW"])) g.markOutput(o);
    g.setScene(scene);
    return g;
}

/** Share of pixels with geometry in G.posW. */
export async function coverage(device: Device, g: RenderGraph): Promise<number> {
    const pos = await read(device, g.getOutput("G.posW")!);
    let hits = 0;
    for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
    return hits / (pos.length / 4);
}

/** Executes a G-buffer-only graph once and checks that the scene shows up. Leaves the camera history as is. */
export async function warmUp(device: Device, scene: Scene): Promise<void> {
    const g = buildGraph(device, scene, [], [], [], 32, 32);
    g.execute(device.renderContext);
    const c = await coverage(device, g);
    if (!(c > 0.2)) throw new Error(`G-buffer empty on the warm-up frame (coverage ${c})`);
}

export function rgbaTexture(device: Device, w: number, h: number, data: Float32Array): Texture {
    return device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, data, ResourceBindFlags.ShaderResource);
}

export interface Pose { pos: V3; target: V3 }
export const kPoseA: Pose = { pos: [2.6, 2.4, 3.4], target: [0, 0.4, -0.3] };
export const kPoseB: Pose = { pos: [-2.5, 2.0, 3.0], target: [0, 0.5, -0.3] };
export const kPoseX: Pose = { pos: [3.2, 1.8, -1.2], target: [0, 0.4, -0.3] };

export function setPose(scene: Scene, p: Pose, shift = 0): void {
    scene.camera.setPosition(new float3(p.pos[0] + shift, p.pos[1], p.pos[2] - 0.5 * shift));
    scene.camera.setTarget(new float3(...p.target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
}

/** Renders `frames` frames at `pose` (camera moving by `step` per frame) and reads `outputs` after each frame. */
export async function renderShot(device: Device, scene: Scene, g: RenderGraph, pose: Pose, frames: number, outputs: string[], step = 0): Promise<Float32Array[][]> {
    const res: Float32Array[][] = [];
    for (let f = 0; f < frames; f++) {
        setPose(scene, pose, step * f);
        g.execute(device.renderContext);
        const row: Float32Array[] = [];
        for (const o of outputs) row.push(await read(device, g.getOutput(o)!));
        res.push(row);
    }
    return res;
}

/**
 * How a shot boundary is signalled before shot X:
 *  camera-reset : Camera.resetHistory();
 *  prime        : one execution of a separate G-buffer graph at X's camera (the "prime" render of a driver), no API call;
 *  explicit     : resetGraphHistory(graph) (the passes' resetHistory()), the camera is left alone;
 *  none         : nothing (negative control: X must differ from the cold render).
 */
export type CutMethod = "camera-reset" | "prime" | "explicit" | "none";

export interface ColdConsecutiveResult { method: CutMethod; equalFrames: number; frames: number; differs: boolean[] }

/**
 * Cold: a new graph on a camera without history (Camera.resetHistory) renders X for `frames` frames.
 * Consecutive: one graph renders A (moving), B (moving), the cut, then X. Each X frame of each output is compared bit for bit.
 */
export async function coldVsConsecutive(device: Device, scene: Scene, mk: () => RenderGraph, outputs: string[], methods: CutMethod[], frames = 3): Promise<ColdConsecutiveResult[]> {
    await warmUp(device, scene);
    setPose(scene, kPoseX);
    scene.camera.resetHistory();
    const cold = await renderShot(device, scene, mk(), kPoseX, frames, outputs);
    const results: ColdConsecutiveResult[] = [];
    for (const method of methods) {
        const g = mk();
        await renderShot(device, scene, g, kPoseA, 4, outputs, 0.05);
        await renderShot(device, scene, g, kPoseB, 4, outputs, -0.04);
        setPose(scene, kPoseX);
        if (method === "camera-reset") scene.camera.resetHistory();
        else if (method === "prime") buildGraph(device, scene, [], [], [], 16, 16).execute(device.renderContext);
        else if (method === "explicit") resetGraphHistory(g);
        const x = await renderShot(device, scene, g, kPoseX, frames, outputs);
        const differs = x.map((row, f) => row.some((buf, o) => !bitwiseEqual(buf, cold[f]![o]!)));
        results.push({ method, equalFrames: differs.filter((d) => !d).length, frames, differs });
    }
    return results;
}
