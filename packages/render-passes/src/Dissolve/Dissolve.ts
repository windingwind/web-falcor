/**
 * Dissolve: per-object dissolve / burn-away with an emissive edge, built on the GBufferRaster per-instance clip.
 *
 * Two passes:
 *  - DissolveControl writes the GBufferRaster `clipBuffer` (one float4 per geometry instance: threshold, noise mode,
 *    edge width, noise scale) for the listed `instances`. Its threshold follows `time`: the dissolve of the instance at
 *    list position j runs over [startTime + j * stagger, startTime + j * stagger + duration] and ends at `threshold`
 *    (progress p = saturate((time - startTime - j * stagger) / duration), 1 when duration = 0; `reverse` uses 1 - p so
 *    the objects appear instead). Noise "object" = object-space value noise (scale `noiseScale`, follows the object),
 *    "dither" = ordered screen-space dither. Unlisted instances are not clipped.
 *  - Dissolve adds the burning edge to the lit colour: with e = GBufferRaster.clipEdge (1 at the cut, 0 beyond
 *    `edgeWidth`), dst = colour * (1 - charAmount * e) + edgeIntensity * lerp(emberColour, edgeColour, e^2) * e^edgeSharpness.
 *    Pixels with e = 0 are copied unchanged.
 * Graph: DissolveControl.clipBuffer -> GBufferRaster.clipBuffer; lit colour (e.g. DeferredLighting.color) -> Dissolve.color;
 * GBufferRaster.clipEdge -> Dissolve.clipEdge.
 * Identity: threshold 0 (or no instances) clips nothing and Dissolve copies its input, bit for bit; edgeIntensity 0 and
 * charAmount 0 leave the lit colour unchanged (only the geometry is cut). Neither pass keeps state: an animation is a
 * pure function of the `time` property (set it per frame).
 *
 * Idea (noise threshold that cuts the surface, emissive border band with a colour ramp) after AdultLink/SphereDissolve:
 * https://github.com/AdultLink/SphereDissolve , MIT License, Copyright (c) 2018 Guillermo Angel. The code here is
 * written for these passes (the clipping lives in GBufferRaster, the border in a compute pass).
 */

import {
    ComputePass,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Dissolve/Dissolve.cs.slang";
const kNoiseModes = { dither: 1, object: 2 } as const;
/** Bytes per clip entry (float4), as read by GBufferRaster. */
const kEntryBytes = 16;

const kControlDefaults = {
    instances: [] as number[],
    threshold: 1,
    time: 0,
    startTime: 0,
    duration: 0,
    stagger: 0,
    reverse: false,
    noise: "object" as string,
    noiseScale: 4,
    edgeWidth: 0.08,
};
type ControlParams = typeof kControlDefaults;

const num = (pass: string, key: string, v: unknown): number => {
    const x = Number(v);
    if (!Number.isFinite(x)) throw new Error(`${pass}: ${key} must be a finite number (got ${JSON.stringify(v)})`);
    return x;
};

/** CPU computation of the clip entries (exported for tests and tools): float4 per instance id 0..max(instances). */
export function dissolveClipEntries(p: ControlParams): Float32Array {
    const ids = p.instances.map((x) => Number(x));
    const n = ids.length ? Math.max(...ids) + 1 : 1;
    const out = new Float32Array(n * 4);
    const mode = kNoiseModes[p.noise as keyof typeof kNoiseModes];
    ids.forEach((id, j) => {
        let prog = p.duration > 0 ? Math.min(Math.max((p.time - p.startTime - j * p.stagger) / p.duration, 0), 1) : 1;
        if (p.reverse) prog = 1 - prog;
        out.set([Math.min(Math.max(p.threshold, 0), 1) * prog, mode, Math.max(p.edgeWidth, 0), p.noiseScale], id * 4);
    });
    return out;
}

export class DissolveControl extends RenderPass {
    private p: ControlParams = { ...kControlDefaults, instances: [] };
    private size = kEntryBytes;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    /** Updates the given properties (others keep their values), e.g. setProperties(new Properties({ time })) per frame. */
    override setProperties(props: Properties): void {
        const next = { ...this.p };
        for (const k of Object.keys(kControlDefaults) as (keyof ControlParams)[]) (next as Record<string, unknown>)[k] = props.get(k, next[k] as never);
        if (!Array.isArray(next.instances) || next.instances.some((x) => !Number.isInteger(Number(x)) || Number(x) < 0 || Number(x) > 1 << 20))
            throw new Error("DissolveControl: instances must be a list of geometry instance ids");
        if (!(next.noise in kNoiseModes)) throw new Error(`DissolveControl: noise must be one of ${Object.keys(kNoiseModes).join(", ")} (got ${next.noise})`);
        for (const k of ["threshold", "time", "startTime", "duration", "stagger", "noiseScale", "edgeWidth"] as const) num("DissolveControl", k, next[k]);
        next.instances = next.instances.map((x) => Number(x));
        this.p = next;
        const size = dissolveClipEntries(this.p).byteLength;
        if (size !== this.size) {
            this.size = size;
            this.requestRecompile();
        }
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Threshold", this.p.threshold, 0, 1, 0.01, (v) => (this.p.threshold = v));
        ui.slider("Time", this.p.time, 0, 60, 0.01, (v) => (this.p.time = v));
        ui.slider("Duration", this.p.duration, 0, 30, 0.01, (v) => (this.p.duration = v));
        ui.slider("Noise scale", this.p.noiseScale, 0.01, 100, 0.01, (v) => (this.p.noiseScale = v));
        ui.slider("Edge width", this.p.edgeWidth, 0, 1, 0.001, (v) => (this.p.edgeWidth = v));
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("clipBuffer", "Per-instance clip entries for GBufferRaster.clipBuffer").rawBuffer(this.size).bindFlags(ResourceBindFlags.ShaderResource);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const buf = rd.getBuffer("clipBuffer");
        if (!buf) return;
        const data = dissolveClipEntries(this.p);
        ctx.updateBuffer(buf, data.subarray(0, Math.floor(buf.size / 4)));
    }
}

const kDefaults = {
    edgeColour: [6, 2.4, 0.6] as number[],
    emberColour: [1.2, 0.15, 0.02] as number[],
    edgeIntensity: 1,
    edgeSharpness: 1.5,
    charAmount: 0.85,
};
type Params = typeof kDefaults;

const vec3 = (key: string, v: unknown): [number, number, number] => {
    if (!Array.isArray(v) || v.length !== 3 || v.some((x) => !Number.isFinite(Number(x)))) throw new Error(`Dissolve: ${key} must be [r, g, b] (got ${JSON.stringify(v)})`);
    return [Number(v[0]), Number(v[1]), Number(v[2])];
};

export class Dissolve extends RenderPass {
    private p: Params = { ...kDefaults };
    private pass: ComputePass | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        this.setProperties(props);
    }

    override setProperties(props: Properties): void {
        const next = { ...this.p };
        for (const k of Object.keys(kDefaults) as (keyof Params)[]) (next as Record<string, unknown>)[k] = props.get(k, next[k] as never);
        vec3("edgeColour", next.edgeColour);
        vec3("emberColour", next.emberColour);
        for (const k of ["edgeIntensity", "edgeSharpness", "charAmount"] as const) num("Dissolve", k, next[k]);
        this.p = next;
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)) as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Edge intensity", this.p.edgeIntensity, 0, 20, 0.01, (v) => (this.p.edgeIntensity = v));
        ui.slider("Edge sharpness", this.p.edgeSharpness, 0.1, 8, 0.01, (v) => (this.p.edgeSharpness = v));
        ui.slider("Char amount", this.p.charAmount, 0, 1, 0.01, (v) => (this.p.charAmount = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "color", "Lit colour");
        r.addInput("clipEdge", "GBufferRaster.clipEdge").format(ResourceFormat.R32Float).bindFlags(ResourceBindFlags.ShaderResource);
        addColorOutput(r, "dst", "Colour with the burning edge", w, h);
        return r;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const dst = rd.getTexture("dst"), color = rd.getTexture("color"), edge = rd.getTexture("clipEdge");
        if (!dst || !color || !edge) return;
        this.pass ??= ComputePass.create(this.device, { path: kShaderFile });
        const root = this.pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [dst.width, dst.height];
        cb["gEdgeColour"] = vec3("edgeColour", this.p.edgeColour);
        cb["gEmberColour"] = vec3("emberColour", this.p.emberColour);
        cb["gEdgeIntensity"] = this.p.edgeIntensity;
        cb["gEdgeSharpness"] = Math.max(this.p.edgeSharpness, 1e-3);
        cb["gCharAmount"] = this.p.charAmount;
        root["gColor"] = color;
        root["gClipEdge"] = edge;
        root["gDst"] = dst;
        this.pass.execute(ctx, dst.width, dst.height);
    }
}

registerRenderPass("DissolveControl", (device, props) => new DissolveControl(device, props), {
    desc: "Time-driven per-object dissolve thresholds for the GBufferRaster clip buffer",
});
registerRenderPass("Dissolve", (device, props) => new Dissolve(device, props), {
    desc: "Burning emissive edge along the GBufferRaster per-instance clip",
});
