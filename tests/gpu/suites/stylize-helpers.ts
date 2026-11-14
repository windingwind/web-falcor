/**
 * Shared fixtures of the stylization pass tests (stylize-*.gpu.test.ts): a small scene (floor + two boxes, three
 * materials, one directional light, optional constant sky), graph building, readback and comparison helpers.
 */

import {
    EnvMap,
    LightType,
    RenderGraph,
    Scene,
    createPass,
    float2,
    float3,
    float4,
    type Device,
    type Texture,
} from "@web-falcor/falcor";

export type V3 = [number, number, number];
export const kSize = 128;
export const kLightDirW: V3 = [0.4, -1, 0.3];
/** Unit vector towards the light. */
export const kL: V3 = (() => {
    const l = Math.hypot(...kLightDirW);
    return kLightDirW.map((x) => -x / l) as V3;
})();
export const kAlbedo: V3[] = [[0.8, 0.8, 0.8], [0.75, 0.12, 0.1], [0.1, 0.25, 0.7]];

function quad(c: V3[], n: V3, materialID: number) {
    const uv = [new float2(0, 0), new float2(1, 0), new float2(1, 1), new float2(0, 1)];
    const t = new float4(1, 0, 0, 1);
    return {
        vertices: c.map((p, i) => ({ position: new float3(...p), normal: new float3(...n), tangent: t, texCrd: uv[i]! })),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
        materialID,
    };
}

/** Closed box (6 faces) as one mesh per face group; all faces share `materialID`. */
function box(min: V3, max: V3, materialID: number) {
    const [x0, y0, z0] = min, [x1, y1, z1] = max;
    return [
        quad([[x0, y1, z1], [x1, y1, z1], [x1, y1, z0], [x0, y1, z0]], [0, 1, 0], materialID),
        quad([[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], [0, 0, 1], materialID),
        quad([[x1, y0, z0], [x0, y0, z0], [x0, y1, z0], [x1, y1, z0]], [0, 0, -1], materialID),
        quad([[x1, y0, z1], [x1, y0, z0], [x1, y1, z0], [x1, y1, z1]], [1, 0, 0], materialID),
        quad([[x0, y0, z0], [x0, y0, z1], [x0, y1, z1], [x0, y1, z0]], [-1, 0, 0], materialID),
    ];
}

export const kBoxA: [V3, V3] = [[-1.2, 0, -0.6], [-0.2, 1.0, 0.4]];
export const kBoxB: [V3, V3] = [[0.3, 0, -1.2], [1.1, 1.6, -0.4]];

/** Object class of a world position: 0 floor, 1 box A, 2 box B, -1 unknown (edges). */
export function objectClass(x: number, y: number, z: number): number {
    const inBox = (b: [V3, V3]) => x >= b[0][0] - 1e-3 && x <= b[1][0] + 1e-3 && y >= b[0][1] - 1e-3 && y <= b[1][1] + 1e-3 && z >= b[0][2] - 1e-3 && z <= b[1][2] + 1e-3;
    const a = inBox(kBoxA), b = inBox(kBoxB);
    if (y < 1e-2 && (a || b)) return -1; // box base on the floor: ambiguous
    if (a) return 1;
    if (b) return 2;
    if (Math.abs(y) < 1e-3) return 0;
    return -1;
}

/** A new test scene (optional constant sky), camera set. */
export function makeScene(device: Device, sky: V3 | null = null, cam: { pos: V3; target: V3 } = { pos: [2.6, 2.4, 3.4], target: [0, 0.4, -0.3] }): Scene {
    const s = 6;
    const meshes = [quad([[-s, 0, s], [s, 0, s], [s, 0, -s], [-s, 0, -s]], [0, 1, 0], 0), ...box(...kBoxA, 1), ...box(...kBoxB, 2)];
    const mats = kAlbedo.map((c) => ({ basic: { baseColor: new float4(c[0], c[1], c[2], 1), specular: new float4(0, 0.6, 0, 0) } }));
    const scene = new Scene(device, meshes, mats, [{ type: LightType.Directional, dirW: new float3(...kLightDirW), intensity: new float3(3, 3, 3) }]);
    if (sky) scene.setEnvMap(new EnvMap(device, { width: 1, height: 1, data: new Float32Array([sky[0], sky[1], sky[2], 1]) } as never));
    scene.camera.setPosition(new float3(...cam.pos));
    scene.camera.setTarget(new float3(...cam.target));
    scene.camera.setUpVector(new float3(0, 1, 0));
    scene.camera.setAspectRatio(1);
    return scene;
}

export type PassDecl = [type: string, id: string, props?: Record<string, unknown>];

let graphCount = 0;

/**
 * Graph = GBufferRaster "G" (+ passes), edges, marked outputs (G.posW is always marked). Executes once, checks that the
 * G-buffer shows the scene, then executes `frames` more times.
 */
export async function runGraph(device: Device, scene: Scene, passes: PassDecl[], edges: [string, string][], outputs: string[], frames = 1): Promise<RenderGraph> {
    const g = new RenderGraph(device, `StylizeTest${++graphCount}`);
    g.onResize(kSize, kSize);
    g.addPass(createPass(device, "GBufferRaster", { samplePattern: "Center" }), "G");
    for (const [type, id, props] of passes) g.addPass(createPass(device, type, props ?? {}), id);
    for (const [a, b] of edges) g.addEdge(a, b);
    for (const o of new Set([...outputs, "G.posW"])) g.markOutput(o);
    g.setScene(scene);
    g.execute(device.renderContext);
    const pos = await read(device, g.getOutput("G.posW")!);
    let hits = 0;
    for (let i = 3; i < pos.length; i += 4) if (pos[i] !== 0) hits++;
    if (hits <= 0.2 * kSize * kSize) throw new Error(`G-buffer empty on the first frame (${hits} px)`);
    for (let i = 0; i < frames; i++) g.execute(device.renderContext);
    return g;
}

export async function read(device: Device, t: Texture): Promise<Float32Array> {
    const raw = await device.renderContext.readTextureSubresource(t);
    return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4).slice(); // own copy (staging memory may be reused)
}

export async function readOut(device: Device, g: RenderGraph, name: string): Promise<Float32Array> {
    const t = g.getOutput(name);
    if (!t) throw new Error(`no output ${name}`);
    return read(device, t);
}

/** Sequential readback of several outputs (one readback in flight at a time). */
export async function readAll(device: Device, g: RenderGraph, names: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = [];
    for (const n of names) out.push((await readOut(device, g, n)).slice());
    return out;
}

/** Per-pixel comparison of rgb against a CPU reference: max abs error and the share of pixels above `tol`. */
export function compareRgb(gpu: Float32Array, ref: (i: number) => V3 | null, tol: number): { maxErr: number; badShare: number; n: number; worst: number } {
    let maxErr = 0, bad = 0, n = 0, worst = -1;
    for (let i = 0; i < kSize * kSize; i++) {
        const r = ref(i);
        if (!r) continue;
        n++;
        const e = Math.max(Math.abs(gpu[i * 4]! - r[0]), Math.abs(gpu[i * 4 + 1]! - r[1]), Math.abs(gpu[i * 4 + 2]! - r[2]));
        if (!(e <= tol)) bad++;
        if (!(e <= maxErr)) { maxErr = e; worst = i; }
    }
    return { maxErr, badShare: bad / Math.max(n, 1), n, worst };
}

/** True when the rgba of two readbacks are bitwise equal on every pixel. */
export function bitwiseEqual(a: Float32Array, b: Float32Array): boolean {
    if (a.length !== b.length) return false;
    const ua = new Uint32Array(a.buffer, a.byteOffset, a.length), ub = new Uint32Array(b.buffer, b.byteOffset, b.length);
    for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
    return true;
}

export const dot3 = (a: ArrayLike<number>, b: ArrayLike<number>) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
export const norm3 = (a: ArrayLike<number>): V3 => {
    const l = Math.hypot(a[0]!, a[1]!, a[2]!) || 1;
    return [a[0]! / l, a[1]! / l, a[2]! / l];
};
export const px = (buf: Float32Array, i: number): [number, number, number, number] => [buf[i * 4]!, buf[i * 4 + 1]!, buf[i * 4 + 2]!, buf[i * 4 + 3]!];
export const clamp01 = (x: number) => Math.min(Math.max(x, 0), 1);
export const lerp3 = (a: ArrayLike<number>, b: ArrayLike<number>, t: number): V3 => [a[0]! + (b[0]! - a[0]!) * t, a[1]! + (b[1]! - a[1]!) * t, a[2]! + (b[2]! - a[2]!) * t];
export const smoothstep = (e0: number, e1: number, x: number) => {
    const t = clamp01((x - e0) / (e1 - e0));
    return t * t * (3 - 2 * t);
};
export const log = (s: string) => console.log(`# ${s}`);

/** Camera data as the passes compute it (StyleCommon.cameraData), from the scene camera. */
export function camInfo(scene: Scene) {
    const c = scene.camera;
    const p = c.getPosition(), t = c.getTarget();
    const fwd = norm3([t.x - p.x, t.y - p.y, t.z - p.z]);
    const v = c.getViewMatrix(), vp = c.getViewProjMatrix();
    const row = (m: typeof v, r: number) => [m.get(r, 0), m.get(r, 1), m.get(r, 2), m.get(r, 3)];
    return { pos: [p.x, p.y, p.z] as V3, fwd, viewRows: [row(v, 0), row(v, 1), row(v, 2)], viewProjRows: [0, 1, 2, 3].map((r) => row(vp, r)), fovY: c.getFovY() };
}

export const srgbEnc = (x: number) => { x = clamp01(x); return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055; };
export const srgbDec = (x: number) => (x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4));
