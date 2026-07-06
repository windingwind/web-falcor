/**
 * Color map pass mirroring Source/RenderPasses/DebugPasses/ColorMapPass:
 * normalizes a channel of the input into [min,max] and applies a color map.
 * Auto-ranging reduces the input with ParallelReduction; like native, the
 * result lands one frame later (frame 0 renders with the static [min,max]).
 */

import {
    Fbo,
    FullScreenPass,
    Logger,
    ParallelReduction,
    ParallelReductionType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/DebugPasses/ColorMapPass/ColorMapPass.ps.slang";

/** Mirrors Falcor::ColorMap (ColorMapParams.slang). */
export enum ColorMap {
    Grey = 0,
    Jet = 1,
    Viridis = 2,
    Plasma = 3,
    Magma = 4,
    Inferno = 5,
}

function formatDefine(format: ResourceFormat): string {
    const name = ResourceFormat[format] ?? "";
    if (name.endsWith("Uint")) return "FORMAT_UINT";
    if (name.endsWith("Sint") || name.endsWith("Int")) return "FORMAT_SINT";
    return "FORMAT_FLOAT";
}

/** Mirrors ColorMapPass::AutoRanging with WebGPU async readback (1+ frame latency). */
class AutoRanging {
    private reduction: ParallelReduction;
    private pending: [number, number] | null = null;
    private inFlight = false;

    constructor(private readonly device: Device) {
        this.reduction = new ParallelReduction(device);
    }

    getMinMax(ctx: RenderContext, texture: Texture, channel: number): [number, number] | null {
        const result = this.pending;
        this.pending = null;
        if (!this.inFlight) {
            this.inFlight = true;
            this.reduction
                .execute(ctx, texture, ParallelReductionType.MinMax)
                .then((v) => {
                    this.pending = [Number(v[channel]), Number(v[4 + channel])];
                    this.inFlight = false;
                })
                .catch((e: unknown) => {
                    Logger.warning(`ColorMapPass auto-range reduction failed: ${String(e)}`);
                    this.inFlight = false;
                });
        }
        return result;
    }
}

export class ColorMapPass extends RenderPass {
    private colorMap = ColorMap.Jet;
    private channel = 0;
    private autoRange = true;
    private minValue = 0;
    private maxValue = 1;
    private autoMinValue = 0;
    private autoMaxValue = 1;
    private autoRanging: AutoRanging | null = null;
    private pass: FullScreenPass | null = null;
    private passKey = "";
    private fbo = new Fbo();

    constructor(device: Device, props: Properties) {
        super(device);
        const map = props.getOpt<string | number>("colorMap");
        if (map !== undefined) this.colorMap = (typeof map === "string" ? ColorMap[map as keyof typeof ColorMap] : map) ?? ColorMap.Jet;
        this.channel = props.get("channel", 0);
        this.autoRange = props.get("autoRange", true);
        this.minValue = props.get("minValue", 0);
        this.maxValue = props.get("maxValue", 1);
    }

    override getProperties(): Properties {
        return new Properties({
            colorMap: ColorMap[this.colorMap]!,
            channel: this.channel,
            autoRange: this.autoRange,
            minValue: this.minValue,
            maxValue: this.maxValue,
        });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("input", "Input image").bindFlags(ResourceBindFlags.ShaderResource).texture2D(0, 0);
        r.addOutput("output", "Output image").bindFlags(ResourceBindFlags.RenderTarget).texture2D(0, 0);
        return r;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const input = renderData.getTexture("input");
        const output = renderData.getTexture("output")!;

        if (this.autoRange && input) {
            this.autoRanging ??= new AutoRanging(this.device);
            const minMax = this.autoRanging.getMinMax(ctx, input, this.channel);
            if (minMax) {
                const [minValue, maxValue] = minMax;
                this.autoMinValue = Math.min(this.autoMinValue, minValue);
                this.autoMaxValue = Math.max(this.autoMaxValue, maxValue);
                const alpha = 0.01;
                this.autoMinValue = this.autoMinValue + (minValue - this.autoMinValue) * alpha;
                this.autoMaxValue = this.autoMaxValue + (maxValue - this.autoMaxValue) * alpha;
                this.minValue = Math.fround(this.autoMinValue);
                this.maxValue = Math.fround(this.autoMaxValue);
            } else {
                this.autoMinValue = this.minValue;
                this.autoMaxValue = this.maxValue;
            }
        } else {
            this.autoRanging = null;
        }

        const fmt = input ? formatDefine(input.format) : "FORMAT_FLOAT";
        const key = `${this.colorMap}:${this.channel}:${fmt}`;
        if (!this.pass || this.passKey !== key) {
            this.pass = FullScreenPass.create(this.device, {
                path: kShaderFile,
                defines: { _COLOR_MAP: this.colorMap, _CHANNEL: this.channel, _FORMAT: fmt },
            });
            this.passKey = key;
        }

        const root = this.pass.getRootVar();
        root["gTexture"] = input!;
        root["StaticCB"]["gParams"]["minValue"] = this.minValue;
        root["StaticCB"]["gParams"]["maxValue"] = this.maxValue;
        this.fbo.attachColorTarget(output, 0);
        this.pass.execute(ctx, this.fbo);
    }
}

registerRenderPass("ColorMapPass", (device, props) => new ColorMapPass(device, props));
