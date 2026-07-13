/**
 * Render pass template mirroring Source/RenderPasses/RenderPassTemplate:
 * the minimal skeleton to copy when authoring a new pass. Passes data
 * through unchanged when both fields connect.
 */

import {
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

export class RenderPassTemplate extends RenderPass {
    constructor(device: Device, _props: Properties) {
        super(device);
    }

    override getProperties(): Properties {
        return new Properties();
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        // Define the required resources here.
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Input data").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("dst", "Output data")
            .texture2D(w, h)
            .format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        // renderData holds the requested resources.
        const src = renderData.getTexture("src");
        const dst = renderData.getTexture("dst");
        if (src && dst) ctx.blit(src, dst);
    }
}

registerRenderPass("RenderPassTemplate", (device, props) => new RenderPassTemplate(device, props));
