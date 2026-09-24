/**
 * Scripted animation vs native (tests/oracle/render-native-anim-script.py): pyscene Animation /
 * addKeyframe / createAnimation / Transform(position, target, up). The light (Hermite + warping)
 * and camera (lookAt keys, Cycle) positions are compared exactly at 7 times; the cube's node
 * (Linear, rotation + scaling, Oscillate) through GBufferRT posW at t=1.3 and 5.2.
 */

import { RenderGraph, createPass, float16ToFloat32, float32ToFloat16, initScripting, runSceneScript, type SceneLight } from "@web-falcor/falcor";
import "@web-falcor/render-passes";
import parseExr from "parse-exr";
import { gpuTest, expectEq } from "../harness/registry.js";

const size = 256;

gpuTest("AnimScript.matchesNative", async ({ device }) => {
    await initScripting("/node_modules/pyodide");
    const base = "/tests/oracle/assets";
    const scene = await runSceneScript(device, await (await fetch(`${base}/oracle-anim-script.pyscene`)).text(), base);
    scene.camera.setAspectRatio(1);
    const record = (await (await fetch("/tests/oracle/out-native/oracle-anim-script.json")).json()) as { time: number; light: number[]; camera: number[]; target: number[] }[];
    let worst = 0;
    for (const r of record) {
        scene.animate(r.time);
        const light = scene.lights[0] as SceneLight;
        const cam = scene.camera;
        const got = [light.position!, cam.getPosition(), cam.getTarget()].flatMap((v) => [v.x, v.y, v.z]);
        const want = [...r.light, ...r.camera, ...r.target];
        const err = Math.max(...got.map((g, i) => Math.abs(g - want[i]!)));
        worst = Math.max(worst, err);
        if (err > 1e-4) console.error(`# anim-script t=${r.time}: got ${got.map((x) => x.toFixed(4))} want ${want.map((x) => x.toFixed(4))}`);
    }
    console.error(`# anim-script: worst light/camera position error ${worst.toExponential(2)}`);
    expectEq(worst < 1e-4, true, `light/camera positions (worst ${worst})`);

    const graph = new RenderGraph(device, "AnimScript");
    graph.addPass(createPass(device, "GBufferRT", {}), "GBufferRT");
    graph.markOutput("GBufferRT.posW");
    graph.onResize(size, size);
    graph.setScene(scene);
    for (const [time, frame] of [[1.3, 13], [5.2, 52]] as const) {
        scene.animate(time);
        graph.execute(device.renderContext);
        const web = new Float32Array((await device.renderContext.readTextureSubresource(graph.getOutput("GBufferRT.posW")!)).buffer);
        const { data, width, height } = parseExr(await (await fetch(`/tests/oracle/out-native/oracle-anim-script.GBufferRT.posW.${frame}.exr`)).arrayBuffer(), 1015) as { data: Float32Array; width: number; height: number };
        let bad = 0;
        for (let y = 0; y < size; y++)
            for (let x = 0; x < size; x++)
                for (let c = 0; c < 3; c++) {
                    const w = float16ToFloat32(float32ToFloat16(web[(y * size + x) * 4 + c]!));
                    const n = data[((height - 1 - y) * width + x) * 4 + c]!;
                    if (Math.abs(w - n) > 1e-2 * Math.max(1, Math.abs(n))) { bad++; break; }
                }
        console.error(`# anim-script posW t=${time}: ${bad} pixels differ`);
        expectEq(bad <= 150, true, `posW pixels ${bad}`);
    }
});
