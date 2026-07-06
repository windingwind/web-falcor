/**
 * Composite pass mirroring Source/RenderPasses/Utils/Composite: blends inputs
 * A and B (add/multiply with scales) into the output. Uses the WebFalcor
 * override shader (write-only output texture, DESIGN.md §4.3).
 */

import {
    ComputePass,
    FieldFlags,
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

const kShaderFile = "RenderPasses/Utils/Composite/Composite.cs.slang";

export enum CompositeMode {
    Add = 0,
    Multiply = 1,
}

/** Mirrors CompositeMode.slangh OUTPUT_FORMAT_*. */
function outputFormatDefine(format: ResourceFormat): number {
    const name = ResourceFormat[format] ?? "";
    if (name.endsWith("Uint")) return 1;
    if (name.endsWith("Sint") || name.endsWith("Int")) return 2;
    return 0;
}

export class Composite extends RenderPass {
    private mode = CompositeMode.Add;
    private scaleA = 1;
    private scaleB = 1;
    private outputFormat = ResourceFormat.RGBA32Float;
    private frameDim: [number, number] = [0, 0];
    private pass: ComputePass | null = null;
    private passKey = "";
    private dummy: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        const mode = props.getOpt<string | number>("mode");
        if (mode !== undefined) this.mode = (typeof mode === "string" ? CompositeMode[mode as keyof typeof CompositeMode] : mode) ?? CompositeMode.Add;
        this.scaleA = props.get("scaleA", 1);
        this.scaleB = props.get("scaleB", 1);
        const fmt = props.getOpt<string | number>("outputFormat");
        if (fmt !== undefined) this.outputFormat = (typeof fmt === "string" ? ResourceFormat[fmt as keyof typeof ResourceFormat] : fmt) ?? this.outputFormat;
    }

    override getProperties(): Properties {
        return new Properties({ mode: CompositeMode[this.mode]!, scaleA: this.scaleA, scaleB: this.scaleB });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addInput("A", "Input A").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addInput("B", "Input B").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("out", "Output").bindFlags(ResourceBindFlags.UnorderedAccess).format(this.outputFormat);
        return r;
    }

    override compile(_ctx: RenderContext, compileData: CompileData): void {
        this.frameDim = compileData.defaultTexDims;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const output = renderData.getTexture("out")!;
        this.outputFormat = output.format;

        const key = `${this.mode}:${outputFormatDefine(this.outputFormat)}`;
        if (!this.pass || this.passKey !== key) {
            this.pass = ComputePass.create(this.device, {
                path: kShaderFile,
                defines: { COMPOSITE_MODE: this.mode, OUTPUT_FORMAT: outputFormatDefine(this.outputFormat) },
            });
            this.passKey = key;
        }

        // Unconnected optional inputs read as zero natively (null SRV); bind a
        // 1x1 black dummy for WebGPU's complete-bind-group requirement.
        this.dummy ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "Composite::dummy",
        });

        const root = this.pass.getRootVar();
        root["CB"]["frameDim"] = this.frameDim;
        root["CB"]["scaleA"] = this.scaleA;
        root["CB"]["scaleB"] = this.scaleB;
        root["A"] = renderData.getTexture("A") ?? this.dummy;
        root["B"] = renderData.getTexture("B") ?? this.dummy;
        root["output"] = output;
        this.pass.execute(ctx, this.frameDim[0], this.frameDim[1]);
    }
}

registerRenderPass("Composite", (device, props) => new Composite(device, props));
