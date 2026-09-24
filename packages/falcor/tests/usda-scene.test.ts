/**
 * USD cameras and lights from layer text (Scene/Importer/UsdaScene.ts), against the
 * conversions of native's USDImporter.
 */
import { describe, expect, it } from "vitest";
import { extractUsdCamerasAndLights, parseUsdaPrims, usdaStageInfo, usdBlackbodyTemperatureAsRgb, usdStageRootTransform } from "../src/Scene/Importer/UsdaScene.js";
import { LightType } from "../src/Scene/SceneData.js";
import { float3 } from "../src/Utils/Math/Vector.js";
import { float4x4, transformPoint, transformVector } from "../src/Utils/Math/Matrix.js";

const close = (a: { x: number; y: number; z: number }, b: number[], eps = 1e-5) => {
    expect(a.x).toBeCloseTo(b[0]!, 5);
    expect(a.y).toBeCloseTo(b[1]!, 5);
    expect(a.z).toBeCloseTo(b[2]!, 5);
    void eps;
};

describe("UsdaScene", () => {
    it("composes xformOps in xformOpOrder through the prim hierarchy", () => {
        const text = `#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "A"
{
    double3 xformOp:translate = (1, 2, 3)
    float3 xformOp:rotateXYZ = (0, 90, 0)
    float3 xformOp:scale = (2, 2, 2)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ", "xformOp:scale"]
    def Xform "B"
    {
        quatf xformOp:orient = (0.70710677, 0, 0, 0.70710677)
        double3 xformOp:translate:pivot = (1, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:translate:pivot", "xformOp:orient", "!invert!xformOp:translate:pivot"]
    }
}
`;
        const [a] = parseUsdaPrims(text, float4x4.identity());
        // translate * rotateY(90) * scale(2): (1, 0, 0) -> (0, 0, -2) + (1, 2, 3).
        close(transformPoint(a!.world, new float3(1, 0, 0)), [1, 2, 1]);
        const b = a!.children[0]!;
        expect(b.path).toBe("/A/B");
        // B rotates 90 degrees about Z around the pivot (1, 0, 0): (1, 1, 0) -> (0, 0, 0) locally.
        close(transformPoint(b.world, new float3(1, 1, 0)), transformPoint(a!.world, new float3(0, 0, 0)).toArray());
    });

    it("applies the stage root transform (meters per unit, Z up) like the native importer", () => {
        const info = usdaStageInfo(`#usda 1.0\n(\n    upAxis = "Z"\n)\n`);
        expect(info).toEqual({ metersPerUnit: 0.01, upAxis: "Z" });
        // scale(0.01), then -90 degrees about X: USD +Z (up) becomes +Y.
        close(transformVector(usdStageRootTransform(info), new float3(0, 0, 100)), [0, 1, 0]);
    });

    it("converts lights, cameras and the dome light", () => {
        const text = `#usda 1.0
(
    metersPerUnit = 1
)
def RectLight "Rect"
{
    float inputs:width = 4
    float inputs:height = 2
    float inputs:intensity = 3
    float inputs:exposure = 1
    color3f inputs:color = (1, 0.5, 0.25)
}
def DiskLight "Disk" { float inputs:radius = 2 }
def DistantLight "Sun"
{
    float inputs:angle = 1
    float3 xformOp:rotateXYZ = (-90, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
}
def DomeLight "Sky"
{
    asset inputs:texture:file = @sky.hdr@
    float inputs:intensity = 2
}
def Camera "Cam"
{
    float focalLength = 24
    float fStop = 2
    float focusDistance = 5
    float2 clippingRange = (0.5, 200)
    float horizontalAperture = 36
    double3 xformOp:translate = (0, 1, 10)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
`;
        const { lights, cameras, domeLight } = extractUsdCamerasAndLights(text);
        const [rect, disk, sun] = lights;
        expect(rect!.type).toBe(LightType.Rect);
        expect(rect!.intensity.toArray()).toEqual([6, 3, 1.5]); // 2^1 * 3 * color
        // Falcor's unit rect is [-1, 1]^2: scale (-w/2, h/2, -1).
        close(transformPoint(rect!.transMat!, new float3(1, 1, 0)), [-2, 1, 0]);
        close(transformVector(disk!.transMat!, new float3(1, 1, 1)), [-2, 2, -1]);
        expect(sun!.type).toBe(LightType.Distant);
        close(sun!.dirW!, [0, -1, 0]); // -Z rotated -90 degrees about X
        expect(sun!.angle).toBeCloseTo((0.5 * Math.PI) / 180, 8);
        expect(domeLight).toMatchObject({ file: "sky.hdr", intensity: 2, tint: [1, 1, 1] });
        const cam = cameras[0]!;
        close(cam.position, [0, 1, 10]);
        close(cam.target, [0, 1, 5]); // focus distance along -Z
        expect(cam.focalLength).toBe(24);
        expect(cam.apertureRadius).toBeCloseTo(0.001 * 0.5 * 24 / 2, 10);
        expect(cam.depthRange).toEqual([0.5, 200]);
        expect(cam.frameWidth).toBe(36);
        expect(cam.frameHeight).toBeUndefined();
    });

    it("matches UsdLuxBlackbodyTemperatureAsRgb", () => {
        // Reference values from the native USD library (clamped to [1000, 10000] K).
        const ref: [number, number[]][] = [
            [500, [4.305504, 0.118358, 0]],
            [1156, [3.946094, 0.225304, 0]],
            [1322, [3.499856, 0.358056, 0]],
            [2750, [1.905274, 0.811147, 0.205075]],
            [4000, [1.414028, 0.924039, 0.533308]],
            [6500, [1.043333, 0.983624, 1.034613]],
            [9999, [0.871857, 0.994688, 1.429953]],
            [12000, [0.871841, 0.994688, 1.429996]],
        ];
        for (const [t, rgb] of ref) {
            const v = usdBlackbodyTemperatureAsRgb(t);
            for (let c = 0; c < 3; c++) expect(v[c]).toBeCloseTo(rgb[c]!, 5);
        }
        const { lights } = extractUsdCamerasAndLights(`#usda 1.0
def SphereLight "S"
{
    bool inputs:enableColorTemperature = true
    float inputs:intensity = 2
}
`);
        close(lights[0]!.intensity, [2 * 1.043333, 2 * 0.983624, 2 * 1.034613]);
    });
});
