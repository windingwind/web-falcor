/**
 * Comparison pass base mirroring Source/RenderPasses/DebugPasses/ComparisonPass:
 * fullscreen split view of two inputs. TextRenderer labels are not ported
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
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

export abstract class ComparisonPass extends RenderPass {
    protected splitShader: FullScreenPass | null = null;
    protected swapSides = false;
    protected splitLoc = -1;
    protected dividerSize = 2;
    protected showLabels = false;
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

        if (this.splitLoc < 0) this.splitLoc = 0.5;
        if (this.showLabels) Logger.warning("ComparisonPass: text labels are not ported (Mogwai UI, M8)");

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
}

/**
 * Mirrors SplitScreenPass: interactive divider comparison. Mouse interaction
 * (hover highlight, arrows, dragging) needs the windowing layer (M8); headless
 * parity covers the no-mouse state (black divider, no arrows).
 */
export class SplitScreenPass extends ComparisonPass {
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

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const root = this.splitShader!.getRootVar();
        // No mouse over the divider in headless runs: unselected black divider,
        // no arrows (kColorUnselected / mDrawArrows && mMouseOverDivider).
        root["GlobalCB"]["gDividerColor"] = [0, 0, 0, 1];
        root["GlobalCB"]["gMousePosition"] = [0, 0];
        root["GlobalCB"]["gDrawArrows"] = 0;
        super.execute(ctx, renderData);
    }
}

registerRenderPass("SideBySidePass", (device, props) => new SideBySidePass(device, props));
registerRenderPass("SplitScreenPass", (device, props) => new SplitScreenPass(device, props));
