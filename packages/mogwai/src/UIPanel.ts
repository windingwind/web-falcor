// DOM implementation of Falcor's UIWidgets + a per-pass controls panel. Each pass's
// renderUI() adds controls here; changing one mutates the pass and fires notify()
// (the viewer resets accumulation). Retained-mode: rebuilt when the graph changes.
import { DomWidgets, type ProgramManager, type RenderGraph, type Scene } from "@web-falcor/falcor";
import { buildScenePanel, type ScenePanelHooks } from "./ScenePanel.js";

/**
 * Rebuilds the controls panel: a "Scene" section (native Scene::renderUI) when a
 * scene is loaded, then one section per pass; passes with no controls are skipped.
 */
export function buildUIPanel(container: HTMLElement, graph: RenderGraph | null, notify: () => void, scene: Scene | null = null, sceneHooks?: ScenePanelHooks, programManager?: ProgramManager): void {
    container.innerHTML = "";
    if (programManager) container.appendChild(frameworkStats(programManager));
    if (scene && sceneHooks) {
        const details = document.createElement("details");
        details.id = "scenePanel";
        const summary = document.createElement("summary");
        summary.textContent = "Scene";
        details.appendChild(summary);
        const body = document.createElement("div");
        details.appendChild(body);
        if (buildScenePanel(body, scene, sceneHooks)) container.appendChild(details);
    }
    if (!graph) return;
    for (const { name, pass } of graph.getPasses()) {
        const details = document.createElement("details");
        details.open = true;
        const summary = document.createElement("summary");
        summary.textContent = name;
        details.appendChild(summary);
        const body = document.createElement("div");
        details.appendChild(body);
        pass.renderUI(new DomWidgets(body, notify));
        if (body.childElementCount > 0) container.appendChild(details);
    }
    if (container.childElementCount === 0) {
        const empty = document.createElement("div");
        empty.className = "ui-text";
        empty.textContent = "(no pass controls)";
        container.appendChild(empty);
    }
}

/** MogwaiSettings' "Framework Stats" group: ProgramManager compilation stats (live while open) and Reset. */
function frameworkStats(pm: ProgramManager): HTMLElement {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Framework Stats";
    const pre = document.createElement("pre");
    pre.className = "ui-text";
    const update = () => {
        const s = pm.getCompilationStats();
        pre.textContent =
            "Program compilation:\n" +
            `Program version count: ${s.programVersionCount}\n` +
            `Program kernels count: ${s.programKernelsCount}\n` +
            `Program version time (total): ${s.programVersionTotalTime.toFixed(3)} s\n` +
            `Program kernels time (total): ${s.programKernelsTotalTime.toFixed(3)} s\n` +
            `Program version time (max): ${s.programVersionMaxTime.toFixed(3)} s\n` +
            `Program kernels time (max): ${s.programKernelsMaxTime.toFixed(3)} s`;
    };
    const reset = Object.assign(document.createElement("button"), { textContent: "Reset" });
    reset.onclick = () => {
        pm.resetCompilationStats();
        update();
    };
    const timer = setInterval(() => (details.isConnected ? details.open && update() : clearInterval(timer)), 1000);
    details.addEventListener("toggle", update);
    details.append(summary, pre, reset);
    update();
    return details;
}
