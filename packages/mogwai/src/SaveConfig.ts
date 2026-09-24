/**
 * Mirrors Mogwai's Renderer::saveConfig: the viewer state as a Mogwai script — every graph
 * (RenderGraphExporter IR + m.addGraph), the scene (m.loadScene + Scene::getScript), the
 * window configuration, the clock and the FrameCapture extension. Loading the script back
 * (viewer ?script= or ScriptRunner) restores the state.
 */

import { ScriptWriter, type Clock, type RenderGraph, type Scene } from "@web-falcor/falcor";
import type { FrameCaptureExtension } from "./FrameCapture.js";

const kRendererVar = "m";

export interface ViewerConfig {
    graphs: RenderGraph[];
    scene: Scene | null;
    /** The scene's path as loaded (m.loadScene argument). */
    scenePath: string | null;
    width: number;
    height: number;
    showUI: boolean;
    clock: Clock;
    frameCapture: FrameCaptureExtension | null;
}

export function saveConfig(c: ViewerConfig): string {
    let s = "";
    if (c.graphs.length > 0) {
        s += "# Graphs\n";
        for (const g of c.graphs) s += g.exportScript();
        s += "\n";
    }
    if (c.scene && c.scenePath) {
        s += "# Scene\n";
        s += ScriptWriter.makeMemberFunc(kRendererVar, "loadScene", ScriptWriter.getPathString(c.scenePath));
        s += c.scene.getScript(`${kRendererVar}.scene`);
        s += "\n";
    }
    s += "# Window Configuration\n";
    s += ScriptWriter.makeMemberFunc(kRendererVar, "resizeFrameBuffer", c.width, c.height);
    s += ScriptWriter.makeSetProperty(kRendererVar, "ui", c.showUI);
    s += "\n# Clock Settings\n";
    s += c.clock.getScript(`${kRendererVar}.clock`) + "\n";
    if (c.frameCapture) s += c.frameCapture.getScript(`${kRendererVar}.${c.frameCapture.getScriptVar()}`) + "\n";
    return s;
}
