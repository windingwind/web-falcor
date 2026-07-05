/**
 * Full-screen pass mirroring Falcor/Core/Pass/FullScreenPass.h.
 * Uses the unmodified upstream vertex shader (Core/Pass/FullScreenPass.vs.slang)
 * with the same quad vertex buffer layout as native Falcor.
 */

import type { Device } from "../API/Device.js";
import type { RenderContext } from "../API/RenderContext.js";
import type { Fbo } from "../API/FBO.js";
import { RasterPass } from "./RasterPass.js";
import { DefineList } from "../Program/DefineList.js";
import { Vao, VertexBufferLayout, VertexLayout, Topology } from "../API/VAO.js";
import { ResourceFormat } from "../API/Formats.js";
import { ResourceBindFlags, MemoryType, ComparisonFunc } from "../API/Types.js";
import { DepthStencilState, DepthStencilStateDesc } from "../API/DepthStencilState.js";
import { CullMode, RasterizerState, RasterizerStateDesc } from "../API/RasterizerState.js";

const kVsPath = "Core/Pass/FullScreenPass.vs.slang";

export interface FullScreenPassDesc {
    /** Pixel shader module path. */
    path: string;
    psEntry?: string;
    defines?: DefineList | Record<string, string | number | boolean>;
}

export class FullScreenPass extends RasterPass {
    static override create(device: Device, desc: FullScreenPassDesc): FullScreenPass {
        return new FullScreenPass(device, desc);
    }

    private constructor(device: Device, desc: FullScreenPassDesc) {
        super(device, {
            path: [kVsPath, desc.path],
            vsEntry: "main",
            psEntry: desc.psEntry ?? "main",
            vsModuleIndex: 0,
            psModuleIndex: 1,
            defines: desc.defines,
        });

        // Quad VB identical to native FullScreenPass: posS float4 + texC float2, triangle strip.
        // NDC corners with y-down texture coords.
        const verts = new Float32Array([
            //  x   y  z  w   u  v
            -1, 1, 0, 1, 0, 0,
            -1, -1, 0, 1, 0, 1,
            1, 1, 0, 1, 1, 0,
            1, -1, 0, 1, 1, 1,
        ]);
        const vb = device.createBuffer(verts.byteLength, ResourceBindFlags.Vertex, MemoryType.DeviceLocal, verts);
        const bufLayout = new VertexBufferLayout();
        bufLayout.addElement("POSITION", 0, ResourceFormat.RGBA32Float, 1, 0);
        bufLayout.addElement("TEXCOORD", 16, ResourceFormat.RG32Float, 1, 1);
        bufLayout.stride = 24;
        const layout = new VertexLayout().addBufferLayout(0, bufLayout);
        this.state.setVao(new Vao(Topology.TriangleStrip, layout, [vb]));
        // Native FullScreenPass: no depth test/write, no culling.
        this.state.setDepthStencilState(
            DepthStencilState.create(new DepthStencilStateDesc().setDepthEnabled(false).setDepthFunc(ComparisonFunc.Always).setDepthWriteMask(false)),
        );
        this.state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
    }

    /** Mirrors FullScreenPass::execute(ctx, fbo). */
    execute(ctx: RenderContext, fbo: Fbo): void {
        this.draw(ctx, fbo, 4);
    }
}
