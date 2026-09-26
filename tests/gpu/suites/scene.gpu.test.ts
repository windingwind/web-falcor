/**
 * M5 Scene GPU smoke test: the host Scene class drives the unmodified upstream
 * Scene.slang module (camera, vertices, instances, materials) end-to-end.
 */

import {
    CameraControllerType,
    ComputePass,
    KeyboardEventType,
    ModifierFlags,
    MouseButton,
    MouseEventType,
    ResourceBindFlags,
    Scene,
    float2,
    float3,
    float4,
} from "@web-falcor/falcor";
import { gpuTest, expectEq, expectClose, expectArrayClose } from "../harness/registry.js";

function makeTriangleScene(device: any): Scene {
    const vertices = [
        { position: new float3(0, 0, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 0) },
        { position: new float3(1, 0, 0), normal: new float3(0, 1, 0), tangent: new float4(1, 0, 0, 1), texCrd: new float2(1, 0) },
        { position: new float3(0, 1, 0), normal: new float3(0, 0, 1), tangent: new float4(1, 0, 0, 1), texCrd: new float2(0, 1) },
    ];
    return new Scene(
        device,
        [{ vertices, indices: new Uint32Array([0, 1, 2]), materialID: 0 }],
        [{ basic: { baseColor: new float4(0.25, 0.5, 0.75, 1.0) } }],
    );
}

gpuTest("Scene.gSceneSmoke", async ({ device }) => {
    const scene = makeTriangleScene(device);
    scene.camera.setPosition(new float3(0, 0, 5));
    scene.camera.setTarget(new float3(0, 0, 0));
    scene.camera.setAspectRatio(1);

    const pass = ComputePass.create(device, { path: "SceneSmoke.cs.slang", defines: scene.getSceneDefines() });
    const out = device.createStructuredBuffer(16, 8, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);

    const root = pass.getRootVar();
    scene.bindShaderData(root);
    root["gOut"] = out;
    pass.execute(device.renderContext, 8);

    const r = new Float32Array((await out.getBlob()).buffer);

    // 0: camera viewProj row 0 == CPU matrix row 0.
    const m = scene.camera.getViewProjMatrix();
    expectArrayClose(r.subarray(0, 4), [m.get(0, 0), m.get(0, 1), m.get(0, 2), m.get(0, 3)], 1e-5, "viewProj row0");

    // 1: vertex 0 position.
    expectArrayClose(r.subarray(4, 8), [0, 0, 0, 1], 1e-6, "vertex0 position");

    // 2: vertex 1 normal (f16 packed round-trip).
    expectArrayClose(r.subarray(8, 12), [0, 1, 0, 0], 1e-2, "vertex1 normal");

    // 3: instance data (materialID=0, vbOffset=0, ibOffset=0, instanceCount=1).
    expectArrayClose(r.subarray(12, 16), [0, 0, 0, 1], 1e-6, "instance data");

    // 4: indices of triangle 0.
    expectArrayClose(r.subarray(16, 20), [0, 1, 2, 0], 1e-6, "triangle indices");

    // 5: material base color (f16 quantized).
    expectArrayClose(r.subarray(20, 24), [0.25, 0.5, 0.75, 1.0], 1e-2, "material baseColor");

    // 6: material header: type Standard(1), IoR 1.5, isBasic 1.
    expectClose(r[24]!, 1, 1e-6, "material type");
    expectClose(r[25]!, 1.5, 1e-2, "IoR");
    expectClose(r[26]!, 1, 1e-6, "isBasicMaterial");

    // 7: world transform (identity) of vertex 0.
    expectArrayClose(r.subarray(28, 32), [0, 0, 0, 1], 1e-6, "world-space vertex");
    out.destroy();
});

gpuTest("Scene.cameraControllerInput", async ({ device }) => {
    // Scene owns the controller (Scene::onKeyEvent/onMouseEvent/onGamepadState + updateSelectedCamera).
    const scene = makeTriangleScene(device);
    const cam = scene.camera;
    cam.setPosition(new float3(0, 0, 5));
    cam.setTarget(new float3(0, 0, 4));
    const key = (k: string, type = KeyboardEventType.KeyPressed, mods = ModifierFlags.None) => scene.onKeyEvent({ type, key: k, mods, codepoint: 0 });
    scene.cameraSpeed = 2;
    scene.updateCamera(0);
    expectEq(key("W"), true, "W handled");
    scene.updateCamera(0.05);
    expectClose(cam.getPosition().z, 5 - 0.1, 1e-5, "moved speed * dt forward");
    key("W", KeyboardEventType.KeyReleased);

    // A handled event stops the camera's animation; C with a modifier restarts it.
    cam.animated = true;
    scene.onMouseEvent({ type: MouseEventType.ButtonDown, pos: [0.5, 0.5], screenPos: [0, 0], wheelDelta: [0, 0], mods: ModifierFlags.None, button: MouseButton.Left });
    expectEq(cam.animated, false, "click stops animation");
    expectEq(key("C", KeyboardEventType.KeyPressed, ModifierFlags.Ctrl), true, "Ctrl+C handled");
    expectEq(cam.animated, true, "Ctrl+C restarts animation");

    // Disabled controls: events are ignored and held keys released.
    key("S");
    scene.setCameraControlsEnabled(false);
    expectEq(key("S"), false, "ignored while disabled");
    expectEq(scene.updateCamera(0.1), false, "held key reset on disable");
    scene.setCameraControlsEnabled(true);

    // F3 adds a viewpoint; the gamepad left stick moves forward.
    const vps = scene.getViewpointCount();
    key("F3");
    expectEq(scene.getViewpointCount(), vps + 1, "F3 adds a viewpoint");
    const z = cam.getPosition().z;
    scene.onGamepadState({ leftX: 0, leftY: -1, rightX: 0, rightY: 0, leftTrigger: -1, rightTrigger: -1, buttons: [] });
    scene.updateCamera(0.15);
    expectClose(cam.getPosition().z, z - 0.1, 1e-5, "gamepad moves speed * dt forward");

    // Orbiter orbits the scene bounds (center, 3.5 radii away), as native setModelParams does.
    scene.setCameraController(CameraControllerType.Orbiter);
    expectEq(scene.getCameraControllerType(), CameraControllerType.Orbiter, "controller type");
    scene.updateCamera(0.2);
    const bb = scene.bounds;
    const p = cam.getPosition(), c = bb.center;
    expectClose(Math.hypot(p.x - c.x, p.y - c.y, p.z - c.z), 3.5 * bb.radius, 1e-4, "orbit distance");
});
