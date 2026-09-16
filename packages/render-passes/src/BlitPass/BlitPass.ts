/**
 * Blit pass mirroring Source/RenderPasses/BlitPass.
 */

import {
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

export class BlitPass extends RenderPass {
    private filter: GPUFilterMode = "linear";

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.filter = props.get<string>("filter", "Linear") === "Point" ? "nearest" : "linear";
    }

    override getProperties(): Properties {
        return new Properties({ filter: this.filter === "nearest" ? "Point" : "Linear" });
    }

    /** Mirrors BlitPass::renderUI. */
    override renderUI(ui: UIWidgets): void {
        ui.dropdown("Filter", ["Linear", "Point"], this.filter === "nearest" ? "Point" : "Linear", (v) => (this.filter = v === "Point" ? "nearest" : "linear"));
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("src", "Source texture").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Destination texture").bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        ctx.blit(renderData.getTexture("src")!, renderData.getTexture("dst")!, this.filter);
    }
}

registerRenderPass("BlitPass", (device, props) => new BlitPass(device, props));
