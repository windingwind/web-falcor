/**
 * Comparison pass base mirroring Source/RenderPasses/DebugPasses/ComparisonPass:
 * fullscreen split view of two inputs. Labels draw through the TextRenderer port
 * (Mogwai UI, M8); the image-test defaults keep them off.
 */

import {
    Fbo,
    FullScreenPass,
    Logger,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    TextRenderer,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

/** Mouse event forwarded by the host (native MouseEvent): `pos` is normalized to the pass output. */
export interface PassMouseEvent {
    type: "buttonDown" | "buttonUp" | "move";
    button?: "left" | "right" | "middle";
    pos: [number, number];
}

export abstract class ComparisonPass extends RenderPass {
    protected splitShader: FullScreenPass | null = null;
    /** Output size of the last execute (native pDstFbo dims, for mouse handling). */
    protected outputDims: [number, number] = [0, 0];
    protected swapSides = false;
    protected splitLoc = -1;
    protected dividerSize = 2;
    protected showLabels = false;
    private textRenderer: TextRenderer | null = null;
    protected leftLabel = "Left side";
    protected rightLabel = "Right side";
    /** SplitScreen assigns the real arrow sprite; others get a 1x1 dummy. */
    protected arrowTex: Texture | null = null;
    private fbo = new Fbo();

    protected parseKeyValuePair(key: string, props: Properties): boolean {
        if (key === "splitLocation") this.splitLoc = props.get("splitLocation", -1);
        else if (key === "showTextLabels") this.showLabels = props.get("showTextLabels", false);
        else if (key === "leftLabel") this.leftLabel = props.get("leftLabel", this.leftLabel);
        else if (key === "rightLabel") this.rightLabel = props.get("rightLabel", this.rightLabel);
        else return false;
        return true;
    }

    override getProperties(): Properties {
        return new Properties({
            splitLocation: this.splitLoc,
            showTextLabels: this.showLabels,
            leftLabel: this.leftLabel,
            rightLabel: this.rightLabel,
        });
    }

    /** Font atlas loads asynchronously (docs §9); wait for it when labels start enabled. */
    override async initAsync(): Promise<void> {
        if (this.showLabels) await this.getTextRenderer().init();
    }

    private getTextRenderer(): TextRenderer {
        return (this.textRenderer ??= new TextRenderer(this.device));
    }

    /** Mirrors ComparisonPass::renderUI. */
    override renderUI(ui: UIWidgets): void {
        ui.checkbox("Swap Sides", this.swapSides, (v) => (this.swapSides = v));
        ui.checkbox("Show Labels", this.showLabels, (v) => (this.showLabels = v));
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("leftInput", "Left side image").bindFlags(ResourceBindFlags.ShaderResource).texture2D(0, 0);
        r.addInput("rightInput", "Right side image").bindFlags(ResourceBindFlags.ShaderResource).texture2D(0, 0);
        r.addOutput("output", "Output image").bindFlags(ResourceBindFlags.RenderTarget).texture2D(0, 0);
        return r;
    }

    /** Subclasses set their pass-specific uniforms, then defer here. */
    override execute(ctx: RenderContext, renderData: RenderData): void {
        const left = renderData.getTexture("leftInput")!;
        const right = renderData.getTexture("rightInput")!;
        const output = renderData.getTexture("output")!;
        this.outputDims = [output.width, output.height];

        if (this.splitLoc < 0) this.splitLoc = 0.5;

        const root = this.splitShader!.getRootVar();
        root["GlobalCB"]["gSplitLocation"] = Math.trunc(this.splitLoc * renderData.defaultTexDims[0]);
        root["GlobalCB"]["gDividerSize"] = this.dividerSize;
        root["gLeftInput"] = this.swapSides ? right : left;
        root["gRightInput"] = this.swapSides ? left : right;
        // gArrowTex is statically reachable in WGSL; native leaves it null when
        // arrows are off (reads return 0 there, and gDrawArrows=false skips it).
        this.arrowTex ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.R8Unorm,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "ComparisonPass::dummyArrow",
        });
        root["gArrowTex"] = this.arrowTex;

        this.fbo.attachColorTarget(output, 0);
        this.splitShader!.execute(ctx, this.fbo);

        // Render some labels (mirrors ComparisonPass::execute; 9 px per monospaced glyph).
        if (this.showLabels) {
            const tr = this.getTextRenderer();
            const screenLocX = Math.trunc(this.splitLoc * renderData.defaultTexDims[0]);
            const screenLocY = Math.trunc(renderData.defaultTexDims[1] - 32);
            const rightSide = this.swapSides ? this.leftLabel : this.rightLabel;
            tr.render(ctx, rightSide, this.fbo, [screenLocX + 16, screenLocY]);
            const leftSide = this.swapSides ? this.rightLabel : this.leftLabel;
            const leftLength = leftSide.length * 9;
            tr.render(ctx, leftSide, this.fbo, [screenLocX - 16 - leftLength, screenLocY]);
        }
    }
}

/** Mirrors SideBySidePass: left cols of each input shown side by side. */
export class SideBySidePass extends ComparisonPass {
    private imageLeftBound = 0;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key] of props.entries()) {
            if (key === "imageLeftBound") this.imageLeftBound = props.get("imageLeftBound", 0);
            else if (!this.parseKeyValuePair(key, props)) Logger.warning(`Unknown property '${key}' in a SideBySidePass properties.`);
        }
        this.splitShader = FullScreenPass.create(device, { path: "RenderPasses/DebugPasses/SideBySidePass/SideBySide.ps.slang" });
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        this.splitShader!.getRootVar()["GlobalCB"]["gLeftBound"] = this.imageLeftBound;
        super.execute(ctx, renderData);
    }

    /** Mirrors SideBySidePass::renderUI (range = half of a 1920-wide output; native uses the live width). */
    override renderUI(ui: UIWidgets): void {
        ui.slider("View Slider", this.imageLeftBound, 0, 960, 1, (v) => (this.imageLeftBound = Math.round(v)));
        super.renderUI(ui);
    }
}

/**
 * Mirrors SplitScreenPass: interactive divider comparison. Mouse interaction
 * (hover highlight, arrows, dragging) needs the windowing layer (M8); headless
 * parity covers the no-mouse state (black divider, no arrows).
 */
export class SplitScreenPass extends ComparisonPass {
    private mouseOverDivider = false;
    private dividerGrabbed = false;
    private mousePos: [number, number] = [0, 0];
    private drawArrows = false;
    private timeOfLastClick = -Infinity;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key] of props.entries()) {
            if (!this.parseKeyValuePair(key, props)) Logger.warning(`Unknown property '${key}' in a SplitScreenPass properties.`);
        }
        this.splitShader = FullScreenPass.create(device, { path: "RenderPasses/DebugPasses/SplitScreenPass/SplitScreen.ps.slang" });
        // 16x16 R8Unorm arrow sprite (kArrowArray in SplitScreenPass.cpp); only
        // mip 0 is ever loaded by the shader.
        const arrow = new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 16,
            height: 16,
            format: ResourceFormat.R8Unorm,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "SplitScreenPass::arrow",
        });
        // prettier-ignore
        arrow.setSubresourceBlob(0, 0, new Uint8Array([
            0,  0,  0,  0,  0,  0,  0,  0,  87, 13, 0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  212,255,255,34, 0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  255,255,255,255,32, 0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  78, 255,255,255,255,33, 0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  81, 255,255,255,255,32, 0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  72, 255,255,255,255,34, 0,
            31, 158,156,156,156,156,156,156,156,146,212,255,255,255,255,34,
            241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,240,
            241,255,255,255,255,255,255,255,255,255,255,255,255,255,255,240,
            31, 158,156,156,156,156,156,156,156,146,212,255,255,255,255,33,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  73, 255,255,255,255,34, 0,
            0,  0,  0,  0,  0,  0,  0,  0,  81, 255,255,255,255,31, 0,  0,
            0,  0,  0,  0,  0,  0,  0,  79, 255,255,255,255,32, 0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  255,255,255,255,31, 0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  212,255,255,33, 0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  87, 12, 0,  0,  0,  0,  0,  0,
        ]));
        this.arrowTex = arrow;
    }

    /** Mirrors SplitScreenPass::renderUI. */
    override renderUI(ui: UIWidgets): void {
        ui.slider("Split location", this.splitLoc < 0 ? 0.5 : this.splitLoc, 0, 1, 0.001, (v) => (this.splitLoc = v));
        ui.checkbox("Show Arrows", this.drawArrows, (v) => (this.drawArrows = v));
        super.renderUI(ui);
    }

    /**
     * Mirrors SplitScreenPass::onMouseEvent: hovering within max(6, dividerSize) px
     * highlights the divider, left-drag moves it, a double click (<100 ms) recenters.
     * Returns true when the event was consumed (the host then skips camera control).
     */
    onMouseEvent(ev: PassMouseEvent): boolean {
        const [w, h] = this.outputDims;
        if (w === 0 || h === 0) return false;
        let handled = this.dividerGrabbed;
        this.mousePos = [
            Math.min(w - 1, Math.max(0, Math.trunc(ev.pos[0] * w))),
            Math.min(h - 1, Math.max(0, Math.trunc(ev.pos[1] * h))),
        ];
        if (this.mouseOverDivider && ev.type === "buttonDown" && ev.button === "left") {
            this.dividerGrabbed = true;
            handled = true;
            const now = performance.now();
            if (now - this.timeOfLastClick < 100) this.splitLoc = 0.5;
            else this.timeOfLastClick = now;
        } else if (this.dividerGrabbed) {
            if (ev.type === "buttonUp" && ev.button === "left") {
                this.dividerGrabbed = false;
                handled = true;
            } else if (ev.type === "move") {
                this.splitLoc = this.mousePos[0] / w;
                handled = true;
            }
        }
        const split = this.splitLoc < 0 ? 0.5 : this.splitLoc;
        this.mouseOverDivider = Math.abs(Math.trunc(split * w) - this.mousePos[0]) < Math.max(6, Math.trunc(this.dividerSize));
        return handled;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const root = this.splitShader!.getRootVar();
        // kColorSelected while the mouse hovers the divider, kColorUnselected otherwise.
        root["GlobalCB"]["gDividerColor"] = this.mouseOverDivider ? [1, 1, 1, 1] : [0, 0, 0, 1];
        root["GlobalCB"]["gMousePosition"] = this.mousePos;
        root["GlobalCB"]["gDrawArrows"] = this.drawArrows && this.mouseOverDivider ? 1 : 0;
        super.execute(ctx, renderData);
    }
}

registerRenderPass("SideBySidePass", (device, props) => new SideBySidePass(device, props));
registerRenderPass("SplitScreenPass", (device, props) => new SplitScreenPass(device, props));
