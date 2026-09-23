/** Mirrors Samples/ShaderToy: a full-screen pixel shader animated by the global clock. */

import { FullScreenPass, SampleApp, type Fbo, type RenderContext, type SampleAppConfig } from "@web-falcor/falcor";

export class ShaderToy extends SampleApp {
    static readonly config: SampleAppConfig = { windowDesc: { width: 1280, height: 720, resizableWindow: true, enableVSync: true, title: "Falcor Shader Toy" } };
    private mainPass: FullScreenPass | null = null;
    private aspectRatio = 0;

    override onLoad(): void {
        this.mainPass = FullScreenPass.create(this.getDevice(), { path: "Samples/ShaderToy/Toy.ps.slang" });
    }

    override onResize(width: number, height: number): void {
        this.aspectRatio = width / height;
    }

    override onFrameRender(renderContext: RenderContext, targetFbo: Fbo): void {
        const cb = this.mainPass!.getRootVar()["ToyCB"];
        cb["iResolution"] = [targetFbo.width, targetFbo.height];
        cb["iGlobalTime"] = this.getGlobalClock().getTime();
        this.mainPass!.execute(renderContext, targetFbo);
    }
}
