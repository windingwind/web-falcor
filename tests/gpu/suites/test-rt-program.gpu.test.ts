/**
 * TestRtProgram (TestPasses): the native shader-binding-table test, lowered to
 * a compute kernel. Oracle: the native hit-group/miss rules evaluated on the
 * CPU against analytic spheres, per camera ray (GBufferRT viewW). Pixels near
 * a silhouette are skipped (the meshes are tessellated spheres).
 */

import { RenderGraph, createPass, initScripting, runSceneScript, type Device, type Scene } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import { gpuTest, expectEq } from "../harness/registry.js";

const W = 256;
const H = 128;
const NX = 4;
const SPACING = 2.5;

type Sphere = { c: [number, number, number]; r: number; meshID: number; userID: number; custom: boolean; masked: boolean };

function spheres(): Sphere[] {
    const out: Sphere[] = [];
    for (let x = 0; x < NX; x++)
        for (let y = 0; y < 4; y++) {
            out.push({ c: [x * SPACING, y * SPACING, 0], r: 1, meshID: x, userID: -1, custom: false, masked: x === 1 || x === 2 });
            out.push({ c: [(NX + x) * SPACING, y * SPACING, 0], r: 0.25 * SPACING, meshID: -1, userID: x, custom: true, masked: false });
        }
    return out;
}

/** Native colour for one pixel, or null when the ray grazes a silhouette. */
function expected(mode: number, px: number, py: number, o: number[], d: number[]): number[] | null {
    const missIndex = (px ^ py) & 0x10 ? 1 : 0;
    const rayType = mode === 0 && (px + py) & 0x8 ? 1 : 0;
    let best: Sphere | null = null;
    let bestT = Infinity;
    for (const s of spheres()) {
        const f = [o[0]! - s.c[0], o[1]! - s.c[1], o[2]! - s.c[2]];
        const b = -(f[0]! * d[0]! + f[1]! * d[1]! + f[2]! * d[2]!);
        const dist = Math.hypot(f[0]! + b * d[0]!, f[1]! + b * d[1]!, f[2]! + b * d[2]!);
        if (Math.abs(dist - s.r) < 0.1 * s.r) return null;
        if (dist > s.r || (mode === 1 && s.custom)) continue;
        // Any-hit runs on masked (non-opaque) meshes only. Mode 0's alpha test never discards
        // them: untextured, it samples the uniform alpha 1. Mode 1 uses the IMtl patterns.
        if (!s.custom && s.masked) {
            if (mode === 1) {
                const type = s.meshID % 3;
                if ((type === 0 && px % 8 === 0) || (type === 1 && py % 8 === 0) || (type === 2 && (px + py) % 8 === 0)) continue;
            }
        }
        if (b < bestT) {
            bestT = b;
            best = s;
        }
    }
    if (!best) return missIndex === 0 ? [0.05, 0.05, 0.05] : [0.1, 0.1, 0.1];
    if (mode === 1) return [[1, 0, 0], [0, 1, 0], [0, 0, 1]][best.meshID % 3]!;
    if (best.custom) {
        if (best.userID % 2 === 0) return rayType === 0 ? [0.75, 0, 1] : [1, 1, 0];
        return rayType === 0 ? [0.25, 0.25, 0.25] : [0.5, 0, 0.5];
    }
    if (best.meshID === 1) return rayType === 0 ? [0, 1, 0] : [1, 0, 0];
    if (best.meshID === 3) return rayType === 0 ? [1, 0, 0] : [0, 1, 0];
    return rayType === 0 ? [0.5, 0.5, 0.5] : [0, 0, 1];
}

async function render(device: Device, scene: Scene, mode: number) {
    const graph = new RenderGraph(device, "TestRtProgram");
    graph.addPass(createPass(device, "TestRtProgram", { mode }), "Test");
    graph.addPass(createPass(device, "GBufferRT", { samplePattern: "Center" }), "GBuffer");
    graph.markOutput("Test.output");
    graph.markOutput("GBuffer.viewW");
    graph.onResize(W, H);
    graph.setScene(scene);
    await graph.init();
    graph.execute(device.renderContext);
    const read = async (name: string) => new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput(name)!)).buffer);
    return { color: await read("Test.output"), viewW: await read("GBuffer.viewW") };
}

for (const mode of [0, 1]) {
    gpuTest(`TestRtProgram.mode${mode}MatchesNativeHitGroups`, async ({ device }) => {
        await initScripting("/node_modules/pyodide");
        const source = await (await fetch("/tests/gpu/assets/testpasses/test-rt-program.pyscene")).text();
        const scene = await runSceneScript(device, source, "/tests/gpu/assets/testpasses");
        scene.camera.setAspectRatio(W / H);
        expectEq([...scene.getMeshIDs()].join(), "0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3", "instances share their native mesh ID");

        const { color, viewW } = await render(device, scene, mode);
        const eye = scene.camera.getPosition();
        const o = [eye.x, eye.y, eye.z];
        let checked = 0;
        let wrong = 0;
        let firstWrong = "";
        const seen = new Set<string>();
        for (let py = 0; py < H; py++)
            for (let px = 0; px < W; px++) {
                const i = (py * W + px) * 4;
                const d = [-viewW[i]!, -viewW[i + 1]!, -viewW[i + 2]!];
                const want = expected(mode, px, py, o, d);
                if (!want) continue;
                checked++;
                const got = [color[i]!, color[i + 1]!, color[i + 2]!];
                seen.add(want.join());
                if (got.some((g, k) => Math.abs(g - want[k]!) > 1e-6) || color[i + 3] !== 1) {
                    wrong++;
                    if (!firstWrong) firstWrong = `(${px},${py}) got ${got.join()} want ${want.join()}`;
                }
            }
        console.error(`# TestRtProgram mode ${mode}: ${checked} pixels checked, ${wrong} wrong ${firstWrong}; ${seen.size} distinct colours`);
        expectEq(checked > W * H * 0.8, true, "most pixels are away from silhouettes");
        expectEq(wrong, 0, "every pixel matches the native hit-group rules");
        expectEq(seen.size, mode === 0 ? 10 : 5, "all hit groups and both miss shaders appear");
    });
}

gpuTest("TestRtProgram.customPrimitiveEdits", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const source = await (await fetch("/tests/gpu/assets/testpasses/test-rt-program.pyscene")).text();
    const scene = await runSceneScript(device, source, "/tests/gpu/assets/testpasses");
    const pass = createPass(device, "TestRtProgram", { mode: 0 }) as unknown as {
        setScene(s: Scene): void;
        addCustomPrimitive(): void;
        moveCustomPrimitive(): void;
        removeCustomPrimitive(i: number): void;
    };
    pass.setScene(scene);
    pass.addCustomPrimitive();
    expectEq(scene.getCustomPrimitiveCount(), 17, "added one");
    const added = scene.getCustomPrimitiveAABB(16);
    const r = (added.max[0] - added.min[0]) / 2;
    expectEq(r >= 0.5 && r <= 1 && added.min[1] + r >= 0 && added.min[1] + r <= 1, true, `random sphere bounds (r = ${r})`);
    pass.moveCustomPrimitive();
    pass.removeCustomPrimitive(16);
    pass.removeCustomPrimitive(99);
    expectEq(scene.getCustomPrimitiveCount(), 16, "removed; out-of-range index ignored");
});
