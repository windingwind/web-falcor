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
