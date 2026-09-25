/**
 * Image loading pass mirroring Source/RenderPasses/ImageLoader.
 * Asset loading is async on the web (docs §9): the texture loads in
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
    decodeExr,
    registerRenderPass,
    RuntimeError,
    IOSize,
    calculateIOSize,
    parseIOSize,
    kMaxPossible,
    type CompileData,
    type Device,
    type RenderContext,
    AssetResolver,
    kProjectMediaUrl,
    type UIWidgets,
} from "@web-falcor/falcor";

/** @deprecated Use AssetResolver search paths (kProjectMediaUrl is the default one). */
export const kMediaBaseUrl = `${kProjectMediaUrl}/`;

export class ImageLoader extends RenderPass {
    private filename = "";
    private srgb = true;
    private generateMips = false;
    private mipLevel = 0;
    private arraySlice = 0;
    private outputSize = IOSize.Default;
    /** Unknown = follow the loaded image (native: the graph's default format). */
    private outputFormat = ResourceFormat.Unknown;
    private texture: Texture | null = null;
    /** Output size/format of the last execute (native mOutputSize/mOutputFormat, for the UI). */
    private lastOutput: { width: number; height: number; format: ResourceFormat } | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        this.filename = props.get("filename", "");
        this.srgb = props.get("srgb", true);
        this.generateMips = props.get("mips", false);
        this.mipLevel = props.get("mipLevel", 0);
        this.arraySlice = props.get("arrayIndex", 0);
        this.outputSize = parseIOSize(props.getOpt("outputSize"));
        const fmt = props.getOpt<string | number>("outputFormat");
        if (fmt !== undefined) this.outputFormat = (typeof fmt === "string" ? ResourceFormat[fmt as keyof typeof ResourceFormat] : fmt) ?? ResourceFormat.Unknown;
    }

    override getProperties(): Properties {
        return new Properties({
            outputSize: IOSize[this.outputSize]!,
            ...(this.outputFormat !== ResourceFormat.Unknown ? { outputFormat: ResourceFormat[this.outputFormat]! } : {}),
            filename: this.filename,
            mips: this.generateMips,
            srgb: this.srgb,
            arrayIndex: this.arraySlice,
            mipLevel: this.mipLevel,
        });
    }

    /** Mirrors ImageLoader::renderUI (no file dialog; sRGB/mips toggles reload asynchronously). */
    override renderUI(ui: UIWidgets): void {
        // Native GBufferBase/ImageLoader/ToneMapper/... "Output size" controls: I/O size changes recompile the graph.
        ui.dropdown("Output size", ["Default", "Fixed", "Full", "Half", "Quarter", "Double"], IOSize[this.outputSize]!, (v) => {
            this.outputSize = IOSize[v as keyof typeof IOSize];
            this.requestRecompile();
        });
        ui.text(`Image File: ${this.filename}`);
        const reload = (set: (v: boolean) => void) => (v: boolean) => {
            set(v);
            void this.initAsync();
        };
        ui.checkbox("Load As SRGB", this.srgb, reload((v) => (this.srgb = v)));
        ui.checkbox("Generate Mipmaps", this.generateMips, reload((v) => (this.generateMips = v)));
        if (this.texture) {
            if (this.texture.mipCount > 1) ui.slider("Mip Level", this.mipLevel, 0, this.texture.mipCount - 1, 1, (v) => (this.mipLevel = Math.round(v)));
            if (this.texture.arraySize > 1) ui.slider("Array Slice", this.arraySlice, 0, this.texture.arraySize - 1, 1, (v) => (this.arraySlice = Math.round(v)));
            ui.text(`Image format: ${ResourceFormat[this.texture.format]}`);
            ui.text(`Image size: (${this.texture.width}, ${this.texture.height})`);
        }
        if (this.lastOutput) {
            ui.text(`Output format: ${ResourceFormat[this.lastOutput.format]}`);
            ui.text(`Output size: (${this.lastOutput.width}, ${this.lastOutput.height})`);
        }
    }

    override async initAsync(): Promise<void> {
        if (!this.filename) throw new RuntimeError("ImageLoader: no filename specified");
        const url = await AssetResolver.getDefaultResolver().resolvePath(this.filename);
        if (!url) throw new RuntimeError(`ImageLoader: Can't find image file '${this.filename}'`);
        const response = await fetch(url);
        if (!response.ok) throw new RuntimeError(`ImageLoader: failed to fetch '${url}' (${response.status})`);

        // Mirrors Texture::createFromFile(path, generateMips, loadAsSrgb).
        const mips = this.generateMips ? kMaxPossible : 1;
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget; // RT: blit-chain mip generation
        if (this.filename.toLowerCase().endsWith(".hdr")) {
            const hdr = decodeHdr(new Uint8Array(await response.arrayBuffer()));
            this.texture = this.device.createTexture2D(hdr.width, hdr.height, ResourceFormat.RGBA32Float, 1, mips, hdr.data, flags);
        } else if (this.filename.toLowerCase().endsWith(".exr")) {
            const exr = decodeExr(await response.arrayBuffer());
            this.texture = this.device.createTexture2D(exr.width, exr.height, ResourceFormat.RGBA32Float, 1, mips, exr.data, flags);
        } else {
            // PNG/JPG via the browser decoder. premultiplyAlpha must be off:
            // native FreeImage loads straight (non-premultiplied) RGBA.
            const bitmap = await createImageBitmap(await response.blob(), { colorSpaceConversion: "none", premultiplyAlpha: "none" });
            const texture = this.device.createTexture2D(
                bitmap.width,
                bitmap.height,
                this.srgb ? ResourceFormat.RGBA8UnormSrgb : ResourceFormat.RGBA8Unorm,
                1,
                mips,
                undefined,
                flags,
            );
            this.device.gpuDevice.queue.copyExternalImageToTexture({ source: bitmap }, { texture: texture.gpuTexture }, [bitmap.width, bitmap.height]);
            this.texture = texture;
        }
        if (this.generateMips) this.texture.generateMips(this.device.renderContext);
        this.texture.name = this.filename;
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        // Default = framebuffer dims (the image is blitted into the output); Fixed = the image's own size.
        const fixed: [number, number] = this.texture ? [this.texture.width, this.texture.height] : [0, 0];
        const [w, h] = calculateIOSize(this.outputSize, fixed, compileData.defaultTexDims);
        // Unknown takes the graph's default format, as natively: Mogwai scripts (8-bit sRGB swapchain)
        // clamp HDR images there; the web viewer's float default keeps them.
        const format = this.outputFormat;
        r.addOutput("dst", "Destination texture")
            .texture2D(w, h)
            .format(format)
            .bindFlags(ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const dst = renderData.getTexture("dst")!;
        this.lastOutput = { width: dst.width, height: dst.height, format: dst.format };
        if (!this.texture) throw new RuntimeError("ImageLoader: initAsync() has not completed");
        // Mirrors the native clamp + single-subresource SRV blit.
        this.mipLevel = Math.min(this.mipLevel, this.texture.mipCount - 1);
        this.arraySlice = Math.min(this.arraySlice, this.texture.arraySize - 1);
        ctx.blit(this.texture, dst, "linear", this.mipLevel, 0, this.arraySlice, 0);
    }
}

registerRenderPass("ImageLoader", (device, props) => new ImageLoader(device, props));
