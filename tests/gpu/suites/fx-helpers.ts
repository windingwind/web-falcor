/**
 * Shared fixtures of the Hi-Z, history, DepthComposite and ParticleLayer tests: graphs of any size with external inputs,
 * camera poses, and the cold-vs-consecutive protocol for passes that keep state across frames.
 */

import {
    RenderGraph,
    ResourceBindFlags,
    ResourceFormat,
    createPass,
    float3,
    type Device,
    type Scene,
    type Texture,
} from "@web-falcor/falcor";
import { resetGraphHistory } from "@web-falcor/render-passes";
import { bitwiseEqual, read, type PassDecl, type V3 } from "./stylize-helpers.js";

let graphCount = 0;

/** GBufferRaster "G" + passes; no execution. */
export function buildGraph(
    device: Device,
    scene: Scene,
    passes: PassDecl[],
    edges: [string, string][],
    outputs: string[],
    w: number,
    h: number,
    inputs: Record<string, Texture> = {},
): RenderGraph {
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

/** Rgba Texture used by fx-helpers. */
export function rgbaTexture(device: Device, w: number, h: number, data: Float32Array): Texture {
    return device.createTexture2D(w, h, ResourceFormat.RGBA32Float, 1, 1, data, ResourceBindFlags.ShaderResource);
}

/** Pose used by fx-helpers. */
export interface Pose {
    pos: V3;
    target: V3;
}
export const kPoseA: Pose = { pos: [2.6, 2.4, 3.4], target: [0, 0.4, -0.3] };
export const kPoseB: Pose = { pos: [-2.5, 2.0, 3.0], target: [0, 0.5, -0.3] };
export const kPoseX: Pose = { pos: [3.2, 1.8, -1.2], target: [0, 0.4, -0.3] };

/** Set Pose used by fx-helpers. */
export function setPose(scene: Scene, p: Pose, shift = 0): void {
    scene.camera.setPosition(new float3(p.pos[0] + shift, p.pos[1], p.pos[2] - 0.5 * shift));
    scene.camera.setTarget(new float3(...p.target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
}

/** Renders `frames` frames at `pose` (camera moving by `step` per frame) and reads `outputs` after each frame. */
export async function renderShot(
    device: Device,
    scene: Scene,
    g: RenderGraph,
    pose: Pose,
    frames: number,
    outputs: string[],
    step = 0,
): Promise<Float32Array[][]> {
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

/** Cold Consecutive Result used by fx-helpers. */
export interface ColdConsecutiveResult {
    method: CutMethod;
    equalFrames: number;
    frames: number;
    differs: boolean[];
    /** X frames whose cold render before the consecutive shots differs from the cold render after them (see below). */
    coldDrift: number;
}

/**
 * Cold: a new graph on a camera without history (Camera.resetHistory) renders X for `frames` frames.
 * Consecutive: one graph renders A (moving), B (moving), the cut, then X. Each X frame of each output is compared bit for bit.
 *
 * The cold shot is rendered twice, before and after all consecutive shots (each time on a new graph and a reset
 * camera), and a consecutive X frame counts as equal when it matches either cold render of that frame bit for bit.
 * Some GPU drivers replace a pipeline's machine code some time after it was created by a re-optimised variant whose
 * floating-point results differ in the last bit; renders before and after that point then differ although no state is
 * involved, e.g. a cold shot right after the first compile and a consecutive shot a second later. Leaked history differs
 * from both cold renders, so the check still catches it; `coldDrift` reports frames where the two cold renders differ.
 */
export async function coldVsConsecutive(
    device: Device,
    scene: Scene,
    mk: () => RenderGraph,
    outputs: string[],
    methods: CutMethod[],
    frames = 3,
): Promise<ColdConsecutiveResult[]> {
    await warmUp(device, scene);
    const renderCold = async () => {
        setPose(scene, kPoseX);
        scene.camera.resetHistory();
        return renderShot(device, scene, mk(), kPoseX, frames, outputs);
    };
    const cold = await renderCold();
    const shots: { method: CutMethod; x: Float32Array[][] }[] = [];
    for (const method of methods) {
        const g = mk();
        await renderShot(device, scene, g, kPoseA, 4, outputs, 0.05);
        await renderShot(device, scene, g, kPoseB, 4, outputs, -0.04);
        setPose(scene, kPoseX);
        if (method === "camera-reset") scene.camera.resetHistory();
        else if (method === "prime") buildGraph(device, scene, [], [], [], 16, 16).execute(device.renderContext);
        else if (method === "explicit") resetGraphHistory(g);
        shots.push({ method, x: await renderShot(device, scene, g, kPoseX, frames, outputs) });
    }
    const coldAfter = await renderCold();
    const same = (a: Float32Array[], b: Float32Array[]) => a.every((buf, o) => bitwiseEqual(buf, b[o]!));
    const coldDrift = cold.filter((row, f) => !same(row, coldAfter[f]!)).length;
    return shots.map(({ method, x }) => {
        const differs = x.map((row, f) => !same(row, cold[f]!) && !same(row, coldAfter[f]!));
        return { method, equalFrames: differs.filter((d) => !d).length, frames, differs, coldDrift };
    });
}
