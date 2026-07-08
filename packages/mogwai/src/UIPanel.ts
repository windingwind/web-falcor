// DOM implementation of Falcor's UIWidgets + a per-pass controls panel. Each pass's
// renderUI() adds controls here; changing one mutates the pass and fires notify()
// (the viewer resets accumulation). Retained-mode: rebuilt when the graph changes.
import type { RenderGraph, UIWidgets } from "@web-falcor/falcor";

class DomWidgets implements UIWidgets {
    constructor(
        private readonly root: HTMLElement,
        private readonly notify: () => void,
    ) {}

    private row(label: string): HTMLElement {
        const row = document.createElement("label");
        row.className = "ui-row";
        const span = document.createElement("span");
        span.textContent = label;
        row.appendChild(span);
        this.root.appendChild(row);
        return row;
    }

    text(label: string): void {
        const div = document.createElement("div");
        div.className = "ui-text";
        div.textContent = label;
        this.root.appendChild(div);
    }

    button(label: string, onClick: () => void): void {
        const btn = document.createElement("button");
        btn.textContent = label;
        btn.onclick = () => {
            onClick();
            this.notify();
        };
        this.root.appendChild(btn);
    }

    checkbox(label: string, value: boolean, onChange: (v: boolean) => void): void {
        const input = document.createElement("input");
        input.type = "checkbox";
        input.checked = value;
        input.onchange = () => {
            onChange(input.checked);
            this.notify();
        };
        this.row(label).appendChild(input);
    }

    slider(label: string, value: number, min: number, max: number, step: number, onChange: (v: number) => void): void {
        const row = this.row(label);
        const out = document.createElement("output");
        out.textContent = value.toFixed(2);
        const input = document.createElement("input");
        input.type = "range";
        input.min = String(min);
        input.max = String(max);
        input.step = String(step);
        input.value = String(value);
        input.oninput = () => {
            const v = Number(input.value);
            out.textContent = v.toFixed(2);
            onChange(v);
            this.notify();
        };
        row.appendChild(input);
        row.appendChild(out);
    }

    dropdown(label: string, options: readonly string[], value: string, onChange: (v: string) => void): void {
        const sel = document.createElement("select");
        for (const o of options) {
            const opt = document.createElement("option");
            opt.value = o;
            opt.textContent = o;
            if (o === value) opt.selected = true;
            sel.appendChild(opt);
        }
        sel.onchange = () => {
            onChange(sel.value);
            this.notify();
        };
        this.row(label).appendChild(sel);
    }

    group(label: string): UIWidgets {
        const details = document.createElement("details");
        details.open = true;
        const summary = document.createElement("summary");
        summary.textContent = label;
        details.appendChild(summary);
        this.root.appendChild(details);
        return new DomWidgets(details, this.notify);
    }
}

/** Rebuilds the per-pass controls panel from the graph; passes with no controls are skipped. */
export function buildUIPanel(container: HTMLElement, graph: RenderGraph | null, notify: () => void): void {
    container.innerHTML = "";
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
