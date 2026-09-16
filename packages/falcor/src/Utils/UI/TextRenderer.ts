/**
 * Text overlay mirroring Falcor/Utils/UI/TextRenderer.{h,cpp}: builds one quad
 * per glyph and draws them with the upstream TextRenderer.3d.slang (alpha
 * blended, no depth). Web divergence (docs §9): the font loads asynchronously —
 * `render()` is a no-op until `Font` is ready (`init()` awaits it).
 */

import type { Device } from "../../Core/API/Device.js";
import type { RenderContext } from "../../Core/API/RenderContext.js";
import type { Fbo } from "../../Core/API/FBO.js";
import { Buffer } from "../../Core/API/Buffer.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { Vao, VertexBufferLayout, VertexLayout, Topology } from "../../Core/API/VAO.js";
import { BlendState, BlendStateDesc, BlendOp, BlendFunc } from "../../Core/API/BlendState.js";
import { DepthStencilState, DepthStencilStateDesc } from "../../Core/API/DepthStencilState.js";
import { RasterizerState, RasterizerStateDesc, CullMode } from "../../Core/API/RasterizerState.js";
import { RasterPass } from "../../Core/Pass/RasterPass.js";
import { Font, kDefaultFontUrl } from "./Font.js";

const kShaderFile = "Utils/UI/TextRenderer.3d.slang";
const kMaxCharCount = 1000;
/** Two triangles per glyph (mirrors kVertexPos). */
const kVertexPos: [number, number][] = [[0, 0], [0, 1], [1, 0], [1, 0], [0, 1], [1, 1]];
/** Vertex: float2 screenPos + float2 texCoord. */
const kVertexStride = 16;

export enum TextRendererFlags {
    None = 0,
    /** Draw a black shadow one pixel down-right first (native default). */
    Shadowed = 1,
}

export class TextRenderer {
    private font: Font | null = null;
    private pass: RasterPass | null = null;
    private vertexBuffer: Buffer | null = null;
    private vao: Vao | null = null;
    private color: [number, number, number] = [1, 1, 1];
    private flags = TextRendererFlags.Shadowed;
    private readonly ready: Promise<void>;

    constructor(
        private readonly device: Device,
        fontUrl = kDefaultFontUrl,
    ) {
        this.ready = Font.createFromFile(device, fontUrl).then((font) => {
            this.font = font;
        });
    }

    /** Resolves once the font atlas is loaded (native constructs synchronously). */
    init(): Promise<void> {
        return this.ready;
    }

    isReady(): boolean {
        return this.font !== null;
    }

    getColor(): [number, number, number] {
        return this.color;
    }

    setColor(color: [number, number, number]): void {
        this.color = color;
    }

    setFlags(flags: TextRendererFlags): void {
        this.flags = flags;
    }

    /** Mirrors TextRenderer::render: optional shadow pass, then the text. */
    render(ctx: RenderContext, text: string, dstFbo: Fbo, pos: [number, number]): void {
        if (!this.font) return;
        if (this.flags & TextRendererFlags.Shadowed) {
            const old = this.color;
            this.color = [0, 0, 0];
            this.renderText(ctx, text, dstFbo, [pos[0] + 1, pos[1] + 1]);
            this.color = old;
        }
        this.renderText(ctx, text, dstFbo, pos);
    }

    private ensurePass(): RasterPass {
        if (this.pass) return this.pass;
        const pass = RasterPass.create(this.device, { path: kShaderFile, vsEntry: "vsMain", psEntry: "psMain" });
        pass.state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc().setDepthEnabled(false)));
        pass.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));

        this.vertexBuffer = new Buffer(this.device, {
            size: kVertexStride * kMaxCharCount * kVertexPos.length,
            bindFlags: ResourceBindFlags.Vertex,
            memoryType: MemoryType.DeviceLocal, // web: upload via setBlob (queue write)
            name: "TextRenderer::vertices",
        });
        const layout = new VertexLayout();
        const vb = new VertexBufferLayout();
        vb.addElement("POSITION", 0, ResourceFormat.RG32Float, 1, 0);
        vb.addElement("TEXCOORD", 8, ResourceFormat.RG32Float, 1, 1);
        vb.stride = kVertexStride;
        layout.addBufferLayout(0, vb);
        this.vao = new Vao(Topology.TriangleList, layout, [this.vertexBuffer]);
        pass.state.setVao(this.vao);
        pass.getRootVar()["gFontTex"] = this.font!.texture;
        this.pass = pass;
        return pass;
    }

    /** Mirrors TextRenderer::setCbData: pixel -> clip transform (y down) + color. */
    private setCbData(dstFbo: Fbo): void {
        const width = dstFbo.width;
        const height = dstFbo.height;
        // Row-major float4x4 like native (mul(M, v)): x' = 2x/w - 1, y' = 1 - 2y/h.
        const vpTransform = [2 / width, 0, 0, -1, 0, -2 / height, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1];
        const cb = this.pass!.getRootVar()["PerFrameCB"];
        cb["gvpTransform"] = vpTransform;
        cb["gFontColor"] = this.color;
    }

    /** Native blends SrcAlpha/OneMinusSrcAlpha; WebGPU needs float32-blendable for 32-bit float targets
     *  (§9: without it glyph quads are drawn opaque). */
    private applyBlendState(dstFbo: Fbo): void {
        const format = dstFbo.getGpuColorFormats()[0] ?? "";
        const blendable = !format.includes("32float") || this.device.hasFeature("float32-blendable");
        const blend = new BlendStateDesc();
        if (blendable) blend.setRtBlend(0, true).setRtParams(0, BlendOp.Add, BlendOp.Add, BlendFunc.SrcAlpha, BlendFunc.OneMinusSrcAlpha, BlendFunc.One, BlendFunc.One);
        this.pass!.state.setBlendState(BlendState.create(blend));
    }

    private renderText(ctx: RenderContext, text: string, dstFbo: Fbo, pos: [number, number]): void {
        const font = this.font!;
        const pass = this.ensurePass();
        this.applyBlendState(dstFbo);
        this.setCbData(dstFbo);
        const chars = Math.min(text.length, kMaxCharCount);
        const verts = new Float32Array(chars * kVertexPos.length * 4);
        const startX = pos[0];
        let [x, y] = pos;
        let vertexCount = 0;
        for (let i = 0; i < chars; i++) {
            const c = text[i]!;
            if (c === "\n") {
                y += font.getFontHeight();
                x = startX;
            } else if (c === "\t") {
                x += font.getTabWidth();
            } else if (c === " ") {
                x += font.getLettersSpacing();
            } else {
                const desc = font.getCharDesc(c);
                for (const [sx, sy] of kVertexPos) {
                    verts[vertexCount * 4] = x + desc.size[0] * sx;
                    verts[vertexCount * 4 + 1] = y + desc.size[1] * sy;
                    verts[vertexCount * 4 + 2] = desc.topLeft[0] + desc.size[0] * sx;
                    verts[vertexCount * 4 + 3] = desc.topLeft[1] + desc.size[1] * sy;
                    vertexCount++;
                }
                x += font.getLettersSpacing();
            }
        }
        if (vertexCount === 0) return;
        // WebGPU queue writes are ordered before the encoded draw (no VAO rotation needed).
        this.vertexBuffer!.setBlob(verts.subarray(0, vertexCount * 4));
        pass.draw(ctx, dstFbo, vertexCount);
    }
}
