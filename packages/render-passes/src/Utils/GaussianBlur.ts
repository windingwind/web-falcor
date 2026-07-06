/**
 * Gaussian blur pass mirroring Source/RenderPasses/Utils/GaussianBlur:
 * separable two-pass fullscreen blur whose output inherits the connected
 * source's format and dimensions (via CompileData::connectedResources).
 * Weights replicate the native float math exactly (Math.fround discipline).
 */

import {
    Buffer,
    Fbo,
    FullScreenPass,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceType,
    RuntimeError,
    Texture,
    TextureAddressingMode,
    TextureFilteringMode,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type Sampler,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/Utils/GaussianBlur/GaussianBlur.ps.slang";
const f = Math.fround;

export class GaussianBlur extends RenderPass {
    private kernelWidth = 5;
    private sigma = 2;
    private ready = false;
    private horizontal: FullScreenPass | null = null;
    private vertical: FullScreenPass | null = null;
    private weights: Buffer | null = null;
    private sampler: Sampler;
    private tmp: Texture | null = null;
    private fbo = new Fbo();
    private tmpFbo = new Fbo();

    constructor(device: Device, props: Properties) {
        super(device);
        this.kernelWidth = props.get("kernelWidth", 5);
        this.sigma = props.get("sigma", 2);
        this.sampler = device.createSampler({
            minFilter: TextureFilteringMode.Linear,
            magFilter: TextureFilteringMode.Linear,
            mipFilter: TextureFilteringMode.Point,
            addressModeU: TextureAddressingMode.Clamp,
            addressModeV: TextureAddressingMode.Clamp,
            addressModeW: TextureAddressingMode.Clamp,
        });
    }

    override getProperties(): Properties {
        return new Properties({ kernelWidth: this.kernelWidth, sigma: this.sigma });
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        this.ready = false;
        const edge = compileData.connectedResources?.getField("src");
        if (edge) {
            const shape = (field: ReturnType<RenderPassReflection["addInput"]>) => {
                field.format(edge.format_);
                field.resourceType = edge.resourceType;
                field.width = edge.width;
                field.height = edge.height;
                field.depth = edge.depth;
                field.sampleCount = edge.sampleCount;
                field.mipCount = edge.mipCount;
                field.arraySize = edge.arraySize;
            };
            shape(r.addInput("src", "input image to be blurred"));
            shape(r.addOutput("dst", "output blurred image"));
            this.ready = true;
        } else {
            r.addInput("src", "input image to be blurred");
            r.addOutput("dst", "output blurred image");
        }
        return r;
    }

    override compile(_ctx: RenderContext, _compileData: CompileData): void {
        if (!this.ready) throw new RuntimeError("GaussianBlur: Missing incoming reflection information");
        const defines: Record<string, string | number> = { _KERNEL_WIDTH: this.kernelWidth };
        this.horizontal = FullScreenPass.create(this.device, { path: kShaderFile, defines: { ...defines, _HORIZONTAL_BLUR: 1 } });
        this.vertical = FullScreenPass.create(this.device, { path: kShaderFile, defines: { ...defines, _VERTICAL_BLUR: 1 } });
        this.updateKernel();
    }

    /** Mirrors GaussianBlur::updateKernel + getCoefficient with C float semantics. */
    private updateKernel(): void {
        const center = Math.floor(this.kernelWidth / 2);
        const coeff = (x: number): number => {
            const sigmaSquared = f(this.sigma * this.sigma);
            const p = f(-f(x * x) / f(2 * sigmaSquared));
            const e = f(Math.exp(p));
            const a = f(f(2 * f(Math.PI)) * sigmaSquared);
            return f(e / a);
        };
        const weights = new Array<number>(center + 1);
        let sum = 0;
        for (let i = 0; i <= center; i++) {
            weights[i] = coeff(i);
            sum = f(sum + (i === 0 ? weights[i]! : f(2 * weights[i]!)));
        }
        const data = new Float32Array(this.kernelWidth);
        for (let i = 0; i <= center; i++) {
            const w = f(weights[i]! / sum);
            data[center + i] = w;
            data[center - i] = w;
        }
        this.weights = new Buffer(this.device, {
            size: this.kernelWidth * 4,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource,
            memoryType: MemoryType.DeviceLocal,
            name: "GaussianBlur::weights",
        });
        this.weights.setBlob(new Uint8Array(data.buffer));
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const src = renderData.getTexture("src")!;
        const dst = renderData.getTexture("dst")!;
        if (!this.horizontal || !this.vertical) throw new RuntimeError("GaussianBlur: compile() has not run");

        if (!this.tmp || this.tmp.width !== src.width || this.tmp.height !== src.height || this.tmp.format !== src.format) {
            this.tmp = new Texture(this.device, {
                type: ResourceType.Texture2D,
                width: src.width,
                height: src.height,
                format: src.format,
                bindFlags: ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource,
                name: "GaussianBlur::tmp",
            });
        }

        this.tmpFbo.attachColorTarget(this.tmp, 0);
        {
            const root = this.horizontal.getRootVar();
            root["gSampler"] = this.sampler;
            root["gSrcTex"] = src;
            root["weights"] = this.weights!;
            this.horizontal.execute(ctx, this.tmpFbo);
        }

        this.fbo.attachColorTarget(dst, 0);
        {
            const root = this.vertical.getRootVar();
            root["gSampler"] = this.sampler;
            root["gSrcTex"] = this.tmp;
            root["weights"] = this.weights!;
            this.vertical.execute(ctx, this.fbo);
        }
    }
}

registerRenderPass("GaussianBlur", (device, props) => new GaussianBlur(device, props));
