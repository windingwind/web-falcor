import { describe, expect, it } from "vitest";
import { Camera } from "../src/Scene/Camera/Camera.js";
import {
    FirstPersonCameraController,
    OrbiterCameraController,
    SixDoFCameraController,
} from "../src/Scene/Camera/CameraController.js";
import { float2, float3, dot3, length3, sub3 } from "../src/Utils/Math/Vector.js";

const mouse = (type: "buttonDown" | "buttonUp" | "move" | "wheel", x: number, y: number, extra: object = {}) =>
    ({ type, pos: new float2(x, y), ...extra }) as const;

function lookNegZ(): Camera {
    const cam = new Camera("test");
    cam.setPosition(new float3(0, 0, 0));
    cam.setTarget(new float3(0, 0, -1));
    cam.setUpVector(new float3(0, 1, 0));
    return cam;
}

describe("OrbiterCameraController", () => {
    it("places the camera on the model sphere along +Z (native setModelParams)", () => {
        const cam = new Camera("test");
        const ctl = new OrbiterCameraController(cam);
        ctl.setModelParams(new float3(1, 2, 3), 2, 1.5);
        expect(ctl.update()).toBe(true);
        expect(ctl.update()).toBe(false); // dirty consumed
        expect(length3(sub3(cam.getTarget(), new float3(1, 2, 3)))).toBeCloseTo(0, 5);
        expect(cam.getPosition().z).toBeCloseTo(3 + 2 * 1.5, 5);
        expect(cam.getPosition().x).toBeCloseTo(1, 5);
        expect(length3(sub3(cam.getUpVector(), new float3(0, 1, 0)))).toBeCloseTo(0, 5);
    });

    it("keeps the orbit radius and unit up-vector through arcball drags", () => {
        const cam = new Camera("test");
        const ctl = new OrbiterCameraController(cam);
        const center = new float3(1, 2, 3);
        ctl.setModelParams(center, 2, 1.5);
        ctl.update();
        const startPos = cam.getPosition();

        ctl.onMouseEvent(mouse("buttonDown", 0.5, 0.5, { button: "left" }));
        ctl.onMouseEvent(mouse("move", 0.62, 0.47));
        ctl.onMouseEvent(mouse("move", 0.7, 0.58));
        expect(ctl.update()).toBe(true);

        expect(length3(sub3(cam.getPosition(), center))).toBeCloseTo(3, 4);
        expect(length3(cam.getUpVector())).toBeCloseTo(1, 5);
        expect(length3(sub3(cam.getPosition(), startPos))).toBeGreaterThan(0.05);
        expect(length3(sub3(cam.getTarget(), new float3(1, 2, 3)))).toBeCloseTo(0, 5);
    });

    it("dollies with the wheel (0.2 radii per notch)", () => {
        const cam = new Camera("test");
        const ctl = new OrbiterCameraController(cam);
        ctl.setModelParams(new float3(0, 0, 0), 2, 1.5);
        ctl.update();
        ctl.onMouseEvent(mouse("wheel", 0.5, 0.5, { wheelDelta: new float2(0, 1) }));
        ctl.update();
        expect(length3(cam.getPosition())).toBeCloseTo(2 * 1.3, 5);
    });
});

describe("FirstPersonCameraController", () => {
    it("moves along the view direction on W (elapsed clamped to 0.1 s)", () => {
        const cam = lookNegZ();
        const ctl = new FirstPersonCameraController(cam);
        ctl.update(0); // arms the timer
        ctl.onKeyEvent({ type: "keyPressed", key: "w" });
        expect(ctl.update(1)).toBe(true);
        expect(cam.getPosition().z).toBeCloseTo(-0.1, 6); // min(0.1, 1) * speed 1
        expect(cam.getTarget().z).toBeCloseTo(-1.1, 6);
        ctl.onKeyEvent({ type: "keyReleased", key: "w" });
        expect(ctl.update(2)).toBe(false);
    });

    it("applies the shift/ctrl speed modifiers", () => {
        const cam = lookNegZ();
        const ctl = new FirstPersonCameraController(cam);
        ctl.update(0);
        ctl.onKeyEvent({ type: "keyPressed", key: "w", shift: true });
        ctl.update(0.01);
        expect(cam.getPosition().z).toBeCloseTo(-0.1, 6); // 0.01 s * 10x
    });

    it("clamps the position to the camera bounds", () => {
        const cam = lookNegZ();
        const ctl = new FirstPersonCameraController(cam);
        ctl.setCameraBounds(new float3(-10, -10, -0.05), new float3(10, 10, 10));
        ctl.update(0);
        ctl.onKeyEvent({ type: "keyPressed", key: "w" });
        ctl.update(1);
        expect(cam.getPosition().z).toBeCloseTo(-0.05, 6);
    });

    it("yaws around the world up on horizontal left-drag (native row-vector rotation)", () => {
        const cam = lookNegZ();
        const ctl = new FirstPersonCameraController(cam);
        ctl.update(0);
        ctl.onMouseEvent(mouse("buttonDown", 0.5, 0.5, { button: "left" }));
        ctl.onMouseEvent(mouse("move", 0.6, 0.5)); // delta (0.1, 0) radians
        expect(ctl.update(0.02)).toBe(true);

        const viewDir = sub3(cam.getTarget(), cam.getPosition());
        expect(viewDir.x).toBeCloseTo(Math.sin(0.1), 5);
        expect(viewDir.z).toBeCloseTo(-Math.cos(0.1), 5);
        expect(length3(cam.getPosition())).toBeCloseTo(0, 5);
        expect(length3(sub3(cam.getUpVector(), new float3(0, 1, 0)))).toBeCloseTo(0, 5); // Y-locked
    });
});

describe("SixDoFCameraController", () => {
    it("rolls the up-vector around the view axis on right-drag", () => {
        const cam = lookNegZ();
        const ctl = new SixDoFCameraController(cam);
        ctl.update(0);
        ctl.onMouseEvent(mouse("buttonDown", 0.5, 0.5, { button: "right" }));
        ctl.onMouseEvent(mouse("move", 0.6, 0.5));
        expect(ctl.update(0.02)).toBe(true);

        const up = cam.getUpVector();
        expect(dot3(up, new float3(0, 1, 0))).toBeCloseTo(Math.cos(0.1), 5);
        const viewDir = sub3(cam.getTarget(), cam.getPosition());
        expect(viewDir.z).toBeCloseTo(-1, 5); // roll leaves the view direction alone
    });
});
