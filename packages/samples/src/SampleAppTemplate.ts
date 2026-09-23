/** Mirrors Samples/SampleAppTemplate: clears the frame and shows a small Gui. */

import { FboAttachmentType, SampleApp, type Fbo, type RenderContext, type SampleAppConfig, type UIWidgets } from "@web-falcor/falcor";

export class SampleAppTemplate extends SampleApp {
    static readonly config: SampleAppConfig = { windowDesc: { title: "Falcor Project Template", resizableWindow: true } };

    override onFrameRender(renderContext: RenderContext, targetFbo: Fbo): void {
        const clearColor: [number, number, number, number] = [0.38, 0.52, 0.1, 1];
        renderContext.clearFbo(targetFbo, clearColor, 1.0, 0, FboAttachmentType.All);
    }

    override onGuiRender(gui: UIWidgets): void {
        const w = gui.group("Falcor");
        this.renderGlobalUI(w);
        w.text("Hello from SampleAppTemplate");
        w.button("Click Here", () => alert("Now why would you do that?"));
    }
}
