/**
 * Camera unit tests: matrix conventions and projective behavior.
 */

import { describe, it, expect } from "vitest";
import { Camera, focalLengthToFovY } from "../src/Scene/Camera/Camera.js";
import { float3, float4 } from "../src/Utils/Math/Vector.js";
import { mulMatVec, transformPoint } from "../src/Utils/Math/Matrix.js";

const closeTo = (a: number, b: number, eps = 1e-4) => expect(Math.abs(a - b)).toBeLessThan(eps);

describe("Camera", () => {
    it("default 21mm focal length gives Falcor's default FOV", () => {
        closeTo(focalLengthToFovY(21, 24), 2 * Math.atan(12 / 21));
    });

    it("projects the target to NDC center", () => {
        const cam = new Camera();
        cam.setPosition(new float3(3, 2, 5));
        cam.setTarget(new float3(-1, 0.5, 0));
        cam.setAspectRatio(16 / 9);
        const clip = mulMatVec(cam.getViewProjMatrix(), new float4(-1, 0.5, 0, 1));
        closeTo(clip.x / clip.w, 0);
        closeTo(clip.y / clip.w, 0);
        expect(clip.z / clip.w).toBeGreaterThan(0);
        expect(clip.z / clip.w).toBeLessThan(1);
    });

    it("view matrix maps camera position to origin", () => {
        const cam = new Camera();
        cam.setPosition(new float3(10, -4, 7));
        cam.setTarget(new float3(0, 0, 0));
        const p = transformPoint(cam.getViewMatrix(), new float3(10, -4, 7));
        closeTo(p.x, 0);
        closeTo(p.y, 0);
        closeTo(p.z, 0);
    });

    it("invViewProj round-trips NDC to world", () => {
        const cam = new Camera();
        cam.setPosition(new float3(0, 1, 4));
        cam.setTarget(new float3(0, 1, 0));
        const world = new float4(0.3, 1.2, -1, 1);
        const clip = mulMatVec(cam.getViewProjMatrix(), world);
        const back = mulMatVec(cam.getData().invViewProj, new float4(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w, 1));
        closeTo(back.x / back.w, world.x, 1e-3);
        closeTo(back.y / back.w, world.y, 1e-3);
        closeTo(back.z / back.w, world.z, 1e-3);
    });

    it("cameraW points from eye to target", () => {
        const cam = new Camera();
        cam.setPosition(new float3(0, 0, 5));
        cam.setTarget(new float3(0, 0, -1));
        const w = cam.getData().cameraW;
        closeTo(w.x, 0);
        closeTo(w.y, 0);
        expect(w.z).toBeLessThan(0);
    });
});
