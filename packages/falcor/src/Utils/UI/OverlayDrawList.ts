/**
 * The ImDrawList subset RenderPass::renderOverlayUI draws with (ImGui::GetBackgroundDrawList),
 * on a 2D canvas over the frame. Geometry follows imgui_draw.cpp: the half-pixel offsets of
 * AddLine/AddRect, radius - 0.5 for stroked circles and n-gons, and nothing drawn for alpha-0 colors.
 */

/** A color as ImColor(float4): RGBA in [0, 1]. */
export type OverlayColor = readonly [number, number, number, number];
type Vec2 = readonly [number, number] | { x: number; y: number };

const xy = (p: Vec2): [number, number] => ("x" in p ? [p.x, p.y] : [p[0], p[1]]);
const byte = (v: number) => Math.round(Math.min(Math.max(v, 0), 1) * 255);
const css = (c: OverlayColor) => `rgba(${byte(c[0])},${byte(c[1])},${byte(c[2])},${byte(c[3]) / 255})`;
const hidden = (c: OverlayColor) => byte(c[3]) === 0;

export class OverlayDrawList {
    /** Native Gui's default font: Trebuchet bold at 14 px. */
    static readonly font = "bold 14px 'Trebuchet MS', sans-serif";

    constructor(readonly ctx: CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D) {}

    private path(points: Vec2[], col: OverlayColor, mode: "stroke" | "closed" | "fill", thickness = 1): void {
        if (hidden(col) || points.length === 0) return;
        const c = this.ctx;
        c.beginPath();
        points.forEach((p, i) => (i === 0 ? c.moveTo(...xy(p)) : c.lineTo(...xy(p))));
        if (mode !== "stroke") c.closePath();
        if (mode === "fill") {
            c.fillStyle = css(col);
            c.fill();
        } else {
            c.strokeStyle = css(col);
            c.lineWidth = thickness;
            c.stroke();
        }
    }

    /** n points of an arc from angle 0 to 2pi(n-1)/n (ImDrawList::PathArcTo as AddNgon uses it). */
    private ngonPoints(center: Vec2, radius: number, n: number): Vec2[] {
        const [cx, cy] = xy(center);
        return Array.from({ length: n }, (_, k) => [cx + Math.cos((2 * Math.PI * k) / n) * radius, cy + Math.sin((2 * Math.PI * k) / n) * radius] as const);
    }

    addLine(p1: Vec2, p2: Vec2, col: OverlayColor, thickness = 1): void {
        const [a, b] = [xy(p1), xy(p2)];
        this.path([[a[0] + 0.5, a[1] + 0.5], [b[0] + 0.5, b[1] + 0.5]], col, "stroke", thickness);
    }

    addRect(pMin: Vec2, pMax: Vec2, col: OverlayColor, thickness = 1): void {
        const [a, b] = [xy(pMin), xy(pMax)];
        const [x0, y0, x1, y1] = [a[0] + 0.5, a[1] + 0.5, b[0] - 0.5, b[1] - 0.5];
        this.path([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], col, "closed", thickness);
    }

    addRectFilled(pMin: Vec2, pMax: Vec2, col: OverlayColor): void {
        const [a, b] = [xy(pMin), xy(pMax)];
        this.path([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]], col, "fill");
    }

    /** Two vertex-colored triangles (UL, UR, BR) and (UL, BR, BL), shaded per pixel like the GPU does. */
    addRectFilledMultiColor(pMin: Vec2, pMax: Vec2, upperLeft: OverlayColor, upperRight: OverlayColor, bottomRight: OverlayColor, bottomLeft: OverlayColor): void {
        if ([upperLeft, upperRight, bottomRight, bottomLeft].every(hidden)) return;
        const [a, b] = [xy(pMin), xy(pMax)];
        const [x0, y0] = [Math.round(a[0]), Math.round(a[1])];
        const [w, h] = [Math.round(b[0]) - x0, Math.round(b[1]) - y0];
        if (w <= 0 || h <= 0) return;
        const img = new ImageData(w, h);
        for (let y = 0; y < h; y++)
            for (let x = 0; x < w; x++) {
                const u = (x + 0.5) / w, v = (y + 0.5) / h;
                // Barycentrics in the triangle this pixel falls in (split along UL-BR).
                const [cUL, cA, cB, wUL, wA, wB] = u >= v ? [upperLeft, upperRight, bottomRight, 1 - u, u - v, v] : [upperLeft, bottomLeft, bottomRight, 1 - v, v - u, u];
                for (let c = 0; c < 4; c++) img.data[(y * w + x) * 4 + c] = byte(cUL[c]! * wUL + cA[c]! * wA + cB[c]! * wB);
            }
        // putImageData ignores compositing, so blend through a scratch canvas.
        const scratch = new OffscreenCanvas(w, h);
        scratch.getContext("2d")!.putImageData(img, 0, 0);
        this.ctx.drawImage(scratch, x0, y0);
    }

    addQuad(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, col: OverlayColor, thickness = 1): void {
        this.path([p1, p2, p3, p4], col, "closed", thickness);
    }

    addQuadFilled(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, col: OverlayColor): void {
        this.path([p1, p2, p3, p4], col, "fill");
    }

    addTriangle(p1: Vec2, p2: Vec2, p3: Vec2, col: OverlayColor, thickness = 1): void {
        this.path([p1, p2, p3], col, "closed", thickness);
    }

    addTriangleFilled(p1: Vec2, p2: Vec2, p3: Vec2, col: OverlayColor): void {
        this.path([p1, p2, p3], col, "fill");
    }

    /** numSegments 0 = a smooth circle (native picks the count from the radius). */
    addCircle(center: Vec2, radius: number, col: OverlayColor, numSegments = 0, thickness = 1): void {
        if (radius < 0.5) return;
        this.path(this.ngonPoints(center, radius - 0.5, numSegments > 0 ? Math.max(numSegments, 3) : 128), col, "closed", thickness);
    }

    addCircleFilled(center: Vec2, radius: number, col: OverlayColor, numSegments = 0): void {
        if (radius < 0.5) return;
        this.path(this.ngonPoints(center, radius, numSegments > 0 ? Math.max(numSegments, 3) : 128), col, "fill");
    }

    addNgon(center: Vec2, radius: number, numSegments: number, col: OverlayColor, thickness = 1): void {
        if (numSegments > 2) this.path(this.ngonPoints(center, radius - 0.5, numSegments), col, "closed", thickness);
    }

    addNgonFilled(center: Vec2, radius: number, numSegments: number, col: OverlayColor): void {
        if (numSegments > 2) this.path(this.ngonPoints(center, radius, numSegments), col, "fill");
    }

    addPolyline(points: Vec2[], col: OverlayColor, closed = false, thickness = 1): void {
        this.path(points, col, closed ? "closed" : "stroke", thickness);
    }

    addConvexPolyFilled(points: Vec2[], col: OverlayColor): void {
        this.path(points, col, "fill");
    }

    addBezierCubic(p1: Vec2, p2: Vec2, p3: Vec2, p4: Vec2, col: OverlayColor, thickness = 1): void {
        if (hidden(col)) return;
        const c = this.ctx;
        c.beginPath();
        c.moveTo(...xy(p1));
        c.bezierCurveTo(...xy(p2), ...xy(p3), ...xy(p4));
        c.strokeStyle = css(col);
        c.lineWidth = thickness;
        c.stroke();
    }

    /** `pos` is the text's top-left corner, as in ImGui. */
    addText(pos: Vec2, col: OverlayColor, text: string): void {
        if (hidden(col) || !text) return;
        const c = this.ctx;
        c.font = OverlayDrawList.font;
        c.textBaseline = "top";
        c.fillStyle = css(col);
        c.fillText(text, ...xy(pos));
    }
}
