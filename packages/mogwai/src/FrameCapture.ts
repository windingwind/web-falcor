/**
 * Mirrors Mogwai/Extensions/Capture/FrameCapture::captureOutput: every mask a
 * graph output was marked with becomes one file. Single channels are copied
 * into a one-channel texture of the same bit depth (".R", ".G", ".B", ".A"),
 * RGB is written as is, and RGBA (".RGBA") exports alpha.
 */

import {
    Bitmap,
    BitmapExportFlags,
    FormatType,
    ImageProcessing,
    Logger,
    ResourceBindFlags,
    ScriptWriter,
    ResourceFormat,
    TextureChannelFlags,
    getFormatChannelCount,
    getFormatType,
    getNumChannelBits,
    toGpuTextureFormat,
    type Device,
    type RenderGraph,
    type Texture,
} from "@web-falcor/falcor";

const kSuffix: Partial<Record<number, [string, number]>> = {
    [TextureChannelFlags.Red]: [".R", 1],
    [TextureChannelFlags.Green]: [".G", 1],
    [TextureChannelFlags.Blue]: [".B", 1],
    [TextureChannelFlags.Alpha]: [".A", 1],
    [TextureChannelFlags.RGB]: ["", 3],
    [TextureChannelFlags.RGBA]: [".RGBA", 4],
};

/** Native's single-channel output format for a source format (Unknown if none). */
export function singleChannelFormat(format: ResourceFormat, mask: number): ResourceFormat {
    const bits = getNumChannelBits(format, Math.log2(mask));
    switch (getFormatType(format)) {
        case FormatType.Unorm:
        case FormatType.UnormSrgb:
            return bits === 8 ? ResourceFormat.R8Unorm : bits === 16 ? ResourceFormat.R16Unorm : ResourceFormat.Unknown;
        case FormatType.Snorm:
            return bits === 8 ? ResourceFormat.R8Snorm : bits === 16 ? ResourceFormat.R16Snorm : ResourceFormat.Unknown;
        case FormatType.Uint:
            return bits === 8 ? ResourceFormat.R8Uint : bits === 16 ? ResourceFormat.R16Uint : bits === 32 ? ResourceFormat.R32Uint : ResourceFormat.Unknown;
        case FormatType.Sint:
            return bits === 8 ? ResourceFormat.R8Int : bits === 16 ? ResourceFormat.R16Int : bits === 32 ? ResourceFormat.R32Int : ResourceFormat.Unknown;
        case FormatType.Float:
            return bits === 16 ? ResourceFormat.R16Float : bits === 32 ? ResourceFormat.R32Float : ResourceFormat.Unknown;
        default:
            return ResourceFormat.Unknown;
    }
}

export interface CapturedFile {
    name: string;
    bytes: Uint8Array;
}

/** Captures graph output `index` for every marked mask; `download` also saves the files in a page. */
export async function captureOutput(device: Device, graph: RenderGraph, index: number, basename: string, download = true): Promise<CapturedFile[]> {
    const outputName = graph.getOutputNames()[index];
    if (!outputName) return [];
    const output = graph.getOutput(outputName);
    if (!output) throw new Error(`Graph output ${outputName} is not a texture`);
    const channels = getFormatChannelCount(output.format);
    const files: CapturedFile[] = [];
    let processing: ImageProcessing | null = null;
    for (const mask of graph.getOutputMasks(index)) {
        const entry = kSuffix[mask];
        if (!entry) {
            Logger.warning(`Graph output ${outputName} mask 0x${mask.toString(16)} is not supported. Skipping.`);
            continue;
        }
        const [suffix, outputChannels] = entry;
        let tex: Texture = output;
        if (outputChannels === 1 && channels > 1) {
            const format = singleChannelFormat(output.format, mask);
            if (format === ResourceFormat.Unknown || toGpuTextureFormat(format) === undefined) {
                Logger.warning(`Graph output ${outputName} mask 0x${mask.toString(16)} failed to determine output format. Skipping.`);
                continue;
            }
            tex = device.createTexture2D(output.width, output.height, format, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
            processing ??= new ImageProcessing(device);
            processing.copyColorChannel(device.renderContext, output, tex, mask);
        }
        const ext = Bitmap.getFileExtFromResourceFormat(tex.format);
        const name = `${basename}${suffix}.${ext}`;
        const flags = mask === TextureChannelFlags.RGBA ? BitmapExportFlags.ExportAlpha : BitmapExportFlags.None;
        files.push({ name, bytes: await tex.captureToFile(0, 0, name, Bitmap.getFormatFromFileExtension(ext), flags, download) });
    }
    return files;
}

/**
 * Mirrors Mogwai's FrameCapture extension (CaptureTrigger + FrameCapture), the
 * `m.frameCapture` object of Mogwai scripts: frames registered with addFrames()
 * are captured at the end of that frame. §9: files go to `captured` (and are
 * downloaded in a page); outputDir only prefixes the recorded path.
 */
export class FrameCaptureExtension {
    outputDir = ".";
    baseFilename = "Mogwai";
    captureAllOutputs = false;
    frameDigits = 0;
    includeOutputInFilename = true;
    outputNameFilter = "";
    ui = false;
    download = true;
    /** Every file captured so far, with its outputDir-relative path. */
    readonly captured: (CapturedFile & { path: string })[] = [];
    private ranges = new Map<RenderGraph, [number, number][]>();
    private current: { graph: RenderGraph; range: [number, number] } | null = null;

    constructor(
        private readonly device: Device,
        private readonly getActiveGraph: () => RenderGraph | null,
        private readonly getGraph: (name: string) => RenderGraph | null,
        private readonly getFrame: () => number,
    ) {}

    /** Mirrors addFrames(graph | name, frames): one single-frame range per entry. */
    addFrames(graph: RenderGraph | string, frames: Iterable<number>): void {
        const g = typeof graph === "string" ? this.getGraph(graph) : graph;
        if (!g) throw new Error(`Can't find a graph named '${String(graph)}'`);
        for (const f of frames) this.addRange(g, Number(f), 1);
    }

    /** Mirrors CaptureTrigger::addRange (count 0 removes the range starting at startFrame). */
    private addRange(graph: RenderGraph, start: number, count: number): void {
        const ranges = this.ranges.get(graph) ?? [];
        this.ranges.set(graph, ranges);
        if (count === 0) {
            const i = ranges.findIndex((r) => r[0] === start);
            if (i >= 0) ranges.splice(i, 1);
            return;
        }
        for (const [s, c] of ranges) {
            if (s === start && c === count) return; // existing ranges are ignored silently
            if (start <= s + c - 1 && s <= start + count - 1) throw new Error("This range overlaps an existing range!");
        }
        ranges.push([start, count]);
    }

    /** Mirrors FrameCapture::getScriptVar. */
    getScriptVar(): string {
        return "frameCapture";
    }

    /** Mirrors FrameCapture::getScript: output settings and the frames registered per graph. */
    getScript(variable: string): string {
        let s = "# Frame Capture\n";
        s += ScriptWriter.makeSetProperty(variable, "outputDir", ScriptWriter.getPathString(this.outputDir));
        s += ScriptWriter.makeSetProperty(variable, "baseFilename", this.baseFilename);
        for (const [graph, ranges] of this.ranges) s += ScriptWriter.makeMemberFunc(variable, "addFrames", graph.name, ranges.map((r) => r[0]));
        return s;
    }

    /** Mirrors reset(graph = None). */
    reset(graph?: RenderGraph | null): void {
        if (graph) this.ranges.delete(graph);
        else this.ranges.clear();
    }

    /** Mirrors print(graph?): the registered start frames. */
    print(graph?: RenderGraph): string {
        const fmt = (g: RenderGraph) => `\tframes = [${(this.ranges.get(g) ?? []).map((r) => r[0]).join(", ")}]`;
        if (graph) return fmt(graph);
        const s = [...this.ranges.keys()].map((g) => `'${g.name}':\n${fmt(g)}\n`).join("");
        return s || "Empty";
    }

    /** Mirrors CaptureTrigger::beginFrame. */
    beginFrame(): void {
        const graph = this.getActiveGraph();
        if (!graph || this.current) return;
        const frame = this.getFrame();
        const range = this.ranges.get(graph)?.find((r) => r[0] === frame);
        if (range) this.current = { graph, range };
    }

    /** Mirrors CaptureTrigger::endFrame: captures while inside a range. */
    async endFrame(): Promise<void> {
        if (!this.current) return;
        const frame = this.getFrame();
        const { graph, range } = this.current;
        if (frame + 1 === range[0] + range[1]) this.current = null;
        await this.triggerFrame(graph, frame);
    }

    /** Mirrors capture(): the active graph's current frame, now. */
    async capture(): Promise<void> {
        const graph = this.getActiveGraph();
        if (graph) await this.triggerFrame(graph, this.getFrame());
    }

    /** Mirrors FrameCapture::triggerFrame. */
    private async triggerFrame(graph: RenderGraph, frame: number): Promise<void> {
        let unmarked: string[] = [];
        if (this.captureAllOutputs) {
            const marked = new Set(graph.getOutputNames());
            unmarked = graph.getAvailableOutputs().filter((o) => !marked.has(o));
            for (const o of unmarked) graph.markOutput(o);
            await graph.init();
            graph.execute(this.device.renderContext);
        }
        const names = graph.getOutputNames();
        for (let i = 0; i < names.length; i++) {
            if (this.outputNameFilter && names[i] !== this.outputNameFilter) continue;
            const frameStr = this.frameDigits > 0 ? String(frame).padStart(this.frameDigits, "0") : String(frame);
            const basename = this.includeOutputInFilename ? `${this.baseFilename}.${names[i]}.${frameStr}` : `${this.baseFilename}_${frameStr}`;
            for (const f of await captureOutput(this.device, graph, i, basename, this.download)) this.captured.push({ ...f, path: `${this.outputDir}/${f.name}` });
        }
        for (const o of unmarked) graph.unmarkOutput(o);
    }
}
