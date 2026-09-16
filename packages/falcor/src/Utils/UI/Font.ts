/**
 * Bitmap font mirroring Falcor/Utils/UI/Font.{h,cpp}: the upstream
 * `dejavu-sans-mono-14.bin` char table + `.dds` glyph atlas, fetched from the
 * served Falcor tree (native reads them from data/framework/fonts).
 */

import type { Device } from "../../Core/API/Device.js";
import { Texture } from "../../Core/API/Texture.js";
import { ResourceBindFlags, ResourceType } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { RuntimeError } from "../../Core/Error.js";
import { decodeDDSToRGBA } from "../../Scene/Importer/DDSLoader.js";

const kFontMagicNumber = 0xdead0001;
const kFirstChar = 0x21; // '!'
const kLastChar = 0x7e; // '~'
const kCharCount = kLastChar - kFirstChar + 1;
const kHeaderSize = 28; // FontFileHeader (packed)
const kCharDataSize = 17; // FontCharData (packed)

/** Default font location (mirrors getRuntimeDirectory()/data/framework/fonts). */
export const kDefaultFontUrl = "/Falcor/data/framework/fonts/dejavu-sans-mono-14";

export interface CharTexCrdDesc {
    /** Top-left texel of the glyph in the atlas. */
    topLeft: [number, number];
    /** Glyph size in texels. */
    size: [number, number];
}

export class Font {
    private constructor(
        readonly texture: Texture,
        private readonly charDesc: CharTexCrdDesc[],
        private readonly fontHeight: number,
        private readonly tabWidth: number,
        private readonly letterSpacing: number,
    ) {}

    /** Mirrors Font::loadFromFile: `<path>.bin` (char table) + `<path>.dds` (atlas). */
    static async createFromFile(device: Device, path = kDefaultFontUrl): Promise<Font> {
        const [binRes, ddsRes] = await Promise.all([fetch(`${path}.bin`), fetch(`${path}.dds`)]);
        if (!binRes.ok || !ddsRes.ok) throw new RuntimeError(`Failed to create font resource '${path}'`);
        const dv = new DataView(await binRes.arrayBuffer());
        const structSize = dv.getUint32(0, true);
        const charDataSize = dv.getUint32(4, true);
        const magic = dv.getUint32(8, true);
        const charCount = dv.getUint32(12, true);
        const fontHeight = dv.getFloat32(16, true);
        const tabWidth = dv.getFloat32(20, true);
        if (structSize !== kHeaderSize || magic !== kFontMagicNumber || charDataSize !== kCharDataSize || charCount !== kCharCount) {
            throw new RuntimeError(`Invalid font file '${path}.bin'`);
        }
        const charDesc: CharTexCrdDesc[] = [];
        let letterSpacing = 0;
        for (let i = 0; i < kCharCount; i++) {
            const o = kHeaderSize + i * kCharDataSize;
            if (dv.getUint8(o) !== kFirstChar + i) throw new RuntimeError(`Font '${path}.bin': unexpected character table`);
            const width = dv.getFloat32(o + 9, true);
            charDesc.push({ topLeft: [dv.getFloat32(o + 1, true), dv.getFloat32(o + 5, true)], size: [width, dv.getFloat32(o + 13, true)] });
            letterSpacing = Math.max(letterSpacing, width);
        }
        // Atlas: uncompressed 32-bit DDS (glyph coverage in alpha; native samples with Load()).
        const { width, height, rgba } = decodeDDSToRGBA(await ddsRes.arrayBuffer(), false, 1 << 16);
        const texture = new Texture(device, {
            type: ResourceType.Texture2D,
            width,
            height,
            format: ResourceFormat.RGBA8Unorm,
            bindFlags: ResourceBindFlags.ShaderResource,
            mipLevels: 1,
            name: "Font::atlas",
        });
        texture.setSubresourceBlob(0, 0, rgba);
        return new Font(texture, charDesc, fontHeight, tabWidth, letterSpacing);
    }

    /** Mirrors Font::getCharDesc (characters outside '!'..'~' map to '?'). */
    getCharDesc(c: string): CharTexCrdDesc {
        const code = c.charCodeAt(0);
        const idx = code >= kFirstChar && code <= kLastChar ? code - kFirstChar : "?".charCodeAt(0) - kFirstChar;
        return this.charDesc[idx]!;
    }

    getFontHeight(): number {
        return this.fontHeight;
    }

    getTabWidth(): number {
        return this.tabWidth;
    }

    /** Advance per character (the widest glyph; the font is monospaced). */
    getLettersSpacing(): number {
        return this.letterSpacing;
    }
}
