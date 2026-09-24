/**
 * OverlaySamplePass mirroring Source/RenderPasses/OverlaySamplePass: copies its input and, in
 * renderOverlayUI, draws a 5x3 grid showing one draw-list primitive per cell over the frame.
 */

import {
    FieldFlags,
    type OverlayDrawList,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type OverlayColor,
    type RenderContext,
} from "@web-falcor/falcor";

type V = [number, number];
const add = (a: V, b: V | number): V => (typeof b === "number" ? [a[0] + b, a[1] + b] : [a[0] + b[0], a[1] + b[1]]);
const sub = (a: V, b: V | number): V => (typeof b === "number" ? [a[0] - b, a[1] - b] : [a[0] - b[0], a[1] - b[1]]);
const mul = (a: V, b: V | number): V => (typeof b === "number" ? [a[0] * b, a[1] * b] : [a[0] * b[0], a[1] * b[1]]);

export class OverlaySamplePass extends RenderPass {
    private frameDim: V = [0, 0];
    private frameCount = 0;

    constructor(device: Device, _props: Properties) {
        super(device);
    }

    override getProperties(): Properties {
        return new Properties();
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("input", "Input buffer").format(ResourceFormat.RGBA32Float).bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("output", "Output buffer of the solution")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("input");
        const dst = renderData.getTexture("output")!;
        this.frameDim = [dst.width, dst.height];
        if (src) ctx.blit(src, dst);
        this.frameCount++;
    }

    override renderOverlayUI(drawList: OverlayDrawList): void {
        const margin = 50;
        const frameMin: V = [margin, margin];
        const frameMax = sub(this.frameDim, margin);
        const frameSize = sub(frameMax, frameMin);
        const white: OverlayColor = [1, 1, 1, 1];

        // Native's frame color has alpha 0, so ImGui skips it; kept as is.
        drawList.addRect(frameMin, frameMax, [1, 0, 0, 0]);

        let primType = 0;
        for (let j = 0; j < 3; j++) {
            for (let i = 0; i < 5; i++) {
                const rectMin = add(add(frameMin, mul(frameSize, [i / 5, j / 3])), margin);
                const rectMax = sub(add(frameMin, mul(frameSize, [(i + 1) / 5, (j + 1) / 3])), margin);
                drawList.addRect(rectMin, rectMax, white);
                const inMin = add(rectMin, margin);
                const inMax = sub(rectMax, margin);
                const center = mul(add(inMin, inMax), 0.5);
                const inSize = sub(inMax, inMin);
                const radius = Math.min(inSize[0], inSize[1]) / 2;
                // The diamonds use the outer rect's center and a size shrunk by 2 margins.
                const diamond = (): V[] => {
                    const c = mul(add(rectMin, rectMax), 0.5);
                    const s = sub(mul(sub(rectMax, rectMin), 0.5), 2 * margin);
                    return [add(c, [0, s[1]]), add(c, [s[0], 0]), add(c, [0, -s[1]]), add(c, [-s[0], 0])];
                };
                const triangle = (): V[] => [add(inMin, mul([0.5, 0], inSize)), add(inMin, inSize), add(inMin, mul([0, 1], inSize))];
                const star = (): V[] =>
                    Array.from({ length: 12 }, (_, k) => {
                        const angle = (2 * 3.14159 * k) / 12;
                        return add(center, mul([Math.cos(angle), Math.sin(angle)], radius * (k % 2 === 0 ? 0.5 : 1)));
                    });
                switch (primType) {
                    case 0: drawList.addLine(inMin, inMax, white); break;
                    case 1: drawList.addRect(inMin, inMax, white); break; // native's "filled rectangle" is an outline too
                    case 2: drawList.addRectFilledMultiColor(inMin, inMax, white, [0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]); break;
                    case 3: { const [p1, p2, p3, p4] = diamond(); drawList.addQuad(p1!, p2!, p3!, p4!, white); break; }
                    case 4: { const [p1, p2, p3, p4] = diamond(); drawList.addQuadFilled(p1!, p2!, p3!, p4!, white); break; }
                    case 5: { const [p1, p2, p3] = triangle(); drawList.addTriangle(p1!, p2!, p3!, white); break; }
                    case 6: { const [p1, p2, p3] = triangle(); drawList.addTriangleFilled(p1!, p2!, p3!, white); break; }
                    case 7: drawList.addCircle(center, radius, white); break;
                    case 8: drawList.addCircleFilled(center, radius, white); break;
                    case 9: drawList.addNgon(center, radius, 5, white); break;
                    case 10: drawList.addNgonFilled(center, radius, 5, white); break;
                    case 11: drawList.addText(inMin, white, "Hello, world!"); break;
                    case 12: drawList.addPolyline(star(), white, false, 1); break;
                    case 13: drawList.addConvexPolyFilled(star(), white); break;
                    case 14:
                        drawList.addBezierCubic(inMin, add(rectMin, mul([0.25, 0.75], sub(rectMax, rectMin))), add(rectMin, mul([0.75, 0.25], sub(rectMax, rectMin))), inMax, white, 1);
                        break;
                }
                primType++;
            }
        }
    }
}

registerRenderPass("OverlaySamplePass", (device, props) => new OverlaySamplePass(device, props));
