/**
 * Mirrors Utils/UI/PixelZoom: while Z is held, a 200x200 point-filtered magnification of the
 * pixels around the mouse is blitted over the back buffer (the mouse wheel changes the zoom).
 */

import type { Device } from "../../Core/API/Device.js";
import type { Fbo } from "../../Core/API/FBO.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import type { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags } from "../../Core/API/Types.js";
import { KeyboardEventType, MouseEventType, type KeyboardEvent, type MouseEvent } from "./InputTypes.js";

function clampToEdge(pix: [number, number], width: number, height: number, offset: number): void {
    const size = [width, height];
    for (let i = 0; i < 2; i++) {
        if (pix[i]! + offset > size[i]!) pix[i] = pix[i]! - (pix[i]! + offset - size[i]!);
        else if (pix[i]! - offset < 0) pix[i] = pix[i]! - (pix[i]! - offset);
    }
}

export class PixelZoom {
    private srcZoomSize = 5;
    private readonly dstZoomSize = 200;
    private readonly zoomCoefficient = 4;
    private srcBlit: Texture | null = null;
    private dstBlit: Texture | null = null;
    private mousePos: [number, number] = [0, 0];
    private shouldZoom = false;

    constructor(
        private readonly device: Device,
        backBuffer: Fbo,
    ) {
        this.onResize(backBuffer);
    }

    onResize(backBuffer: Fbo): void {
        const color = backBuffer.getColorTexture(0)!;
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget;
        this.srcBlit = this.device.createTexture2D(color.width, color.height, color.format, 1, 1, undefined, flags);
        this.dstBlit ??= this.device.createTexture2D(this.dstZoomSize, this.dstZoomSize, color.format, 1, 1, undefined, flags);
    }

    render(ctx: RenderContext, backBuffer: Fbo): void {
        if (!this.shouldZoom) return;
        const color = backBuffer.getColorTexture(0)!;
        const [w, h] = [color.width, color.height];
        // Copy the back buffer, blit the block around the mouse into the zoom texture, and that back over the frame.
        ctx.copyTexture(this.srcBlit!, color);
        let offset = Math.floor(this.srcZoomSize / 2);
        const srcPix: [number, number] = [this.mousePos[0] * w, this.mousePos[1] * h];
        clampToEdge(srcPix, w, h, offset);
        const zoomRect = [0, 0, this.dstZoomSize, this.dstZoomSize];
        ctx.blit(this.srcBlit!, this.dstBlit!, "nearest", 0, 0, 0, 0, undefined, { srcRect: [srcPix[0] - offset, srcPix[1] - offset, srcPix[0] + offset, srcPix[1] + offset].map(Math.floor), dstRect: zoomRect });
        offset = this.dstZoomSize / 2;
        clampToEdge(srcPix, w, h, offset);
        ctx.blit(this.dstBlit!, color, "nearest", 0, 0, 0, 0, undefined, { srcRect: zoomRect, dstRect: [srcPix[0] - offset, srcPix[1] - offset, srcPix[0] + offset, srcPix[1] + offset].map(Math.floor) });
    }

    onMouseEvent(me: MouseEvent): boolean {
        if (!this.shouldZoom) return false;
        this.mousePos = [me.pos[0], me.pos[1]];
        // Scrolling up zooms in.
        const zoomDelta = -this.zoomCoefficient * Math.trunc(me.wheelDelta[1]);
        this.srcZoomSize = Math.max(this.srcZoomSize + zoomDelta, 3);
        return me.type !== MouseEventType.Move; // other handlers still see mouse movement
    }

    onKeyboardEvent(ke: KeyboardEvent): boolean {
        if ((ke.type === KeyboardEventType.KeyPressed || ke.type === KeyboardEventType.KeyReleased) && ke.key === "Z") {
            this.shouldZoom = ke.type === KeyboardEventType.KeyPressed;
            return true;
        }
        return false;
    }
}
