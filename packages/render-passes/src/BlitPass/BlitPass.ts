/**
 * Blit pass mirroring Source/RenderPasses/BlitPass.
 */

import {
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

export class BlitPass extends RenderPass {
    private mFilter: GPUFilterMode = "linear";

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.mFilter = props.get<string>("filter", "Linear") === "Point" ? "nearest" : "linear";
    }

    /** Python `filter`: enumToString/stringToEnum over TextureFilteringMode ("Point", "Linear"). */
    get filter(): string {
        return this.mFilter === "nearest" ? "Point" : "Linear";
    }
    set filter(value: string) {
        if (value !== "Point" && value !== "Linear") throw new RuntimeError(`Invalid enum name '${value}'`);
        this.mFilter = value === "Point" ? "nearest" : "linear";
    }

    override getProperties(): Properties {
        return new Properties({ filter: this.mFilter === "nearest" ? "Point" : "Linear" });
    }

    /** Mirrors BlitPass::renderUI. */
    override renderUI(ui: UIWidgets): void {
        ui.dropdown("Filter", ["Linear", "Point"], this.mFilter === "nearest" ? "Point" : "Linear", (v) => (this.mFilter = v === "Point" ? "nearest" : "linear"));
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("src", "Source texture").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Destination texture").bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        ctx.blit(renderData.getTexture("src")!, renderData.getTexture("dst")!, this.mFilter);
    }
}

registerRenderPass("BlitPass", (device, props) => new BlitPass(device, props));
