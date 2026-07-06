/**
 * Image loading pass mirroring Source/RenderPasses/ImageLoader.
 * Asset loading is async on the web (DESIGN.md §9): the texture loads in
 * initAsync, resolved against the media base URL (AssetResolver-lite).
 */

import {
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    Texture,
    decodeHdr,
    registerRenderPass,
    RuntimeError,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

/** Media search base (mirrors Falcor's data directories). */
export const kMediaBaseUrl = "/Falcor/media/";

export class ImageLoader extends RenderPass {
    private filename = "";
    private srgb = true;
    private texture: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.filename = props.get("filename", "");
        this.srgb = props.get("srgb", true);
        // 'mips', 'outputFormat', 'arraySlice', 'mipLevel' accepted (mip generation lands with TextureManager).
    }

    override getProperties(): Properties {
        return new Properties({ filename: this.filename, srgb: this.srgb });
    }

    override async initAsync(): Promise<void> {
        if (!this.filename) throw new RuntimeError("ImageLoader: no filename specified");
        const url = kMediaBaseUrl + this.filename;
        const response = await fetch(url);
        if (!response.ok) throw new RuntimeError(`ImageLoader: failed to fetch '${url}' (${response.status})`);

        if (this.filename.toLowerCase().endsWith(".hdr")) {
            const hdr = decodeHdr(new Uint8Array(await response.arrayBuffer()));
            this.texture = this.device.createTexture2D(hdr.width, hdr.height, ResourceFormat.RGBA32Float, 1, 1, hdr.data);
        } else {
            // PNG/JPG via the browser decoder. premultiplyAlpha must be off:
            // native FreeImage loads straight (non-premultiplied) RGBA.
            const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
            const texture = this.device.createTexture2D(
                bitmap.width,
                bitmap.height,
                this.srgb ? ResourceFormat.RGBA8UnormSrgb : ResourceFormat.RGBA8Unorm,
                1,
                1,
                undefined,
                ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget,
            );
            this.device.gpuDevice.queue.copyExternalImageToTexture({ source: bitmap }, { texture: texture.gpuTexture }, [bitmap.width, bitmap.height]);
            this.texture = texture;
        }
        this.texture.name = this.filename;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        // Native default IO size selection = framebuffer dims (the loaded image
        // is blitted into the output), NOT the image's own size.
        const [w, h] = compileData.defaultTexDims;
        r.addOutput("dst", "Loaded image")
            .texture2D(w, h)
            .format(this.texture?.format ?? ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.texture) throw new RuntimeError("ImageLoader: initAsync() has not completed");
        ctx.blit(this.texture, renderData.getTexture("dst")!);
    }
}

registerRenderPass("ImageLoader", (device, props) => new ImageLoader(device, props));
