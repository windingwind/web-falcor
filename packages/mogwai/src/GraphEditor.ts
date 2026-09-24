// Render-graph editor panel (DOM/SVG analog of Falcor's RenderGraphEditor): passes laid out by
// dependency depth with input/output ports, edges as curves, and the graph's editing API
// (addPass/removePass, addEdge/removeEdge, markOutput/unmarkOutput) wired to clicks. Nodes can be
// dragged; clicking a node shows its pass UI (renderUI) under the canvas, like the editor's
// node properties window.
import { DomWidgets, createPass, getRegisteredRenderPasses, type RenderGraph, type RenderGraphEdge, type RenderPass } from "@web-falcor/falcor";

export interface GraphEditorHooks {
    /** Called after any edit (viewer: refresh outputs, rebuild pass panels, restart accumulation). */
    onGraphChanged: () => void;
    /** Graph output dimensions (reflection needs them). */
    defaultTexDims: () => [number, number];
    /** Called when the selected pass's UI changes a property (viewer: restart accumulation). */
    onPassPropertiesChanged?: () => void;
}

type Edge = RenderGraphEdge;

const kNodeW = 170;
const kPortH = 16;
const kColGap = 60;
const kRowGap = 24;

/** Graph edges as fresh copies (RenderGraph.getEdges(), addEdge order). */
export function graphEdges(graph: RenderGraph): Edge[] {
    return graph.getEdges().map((e) => ({ ...e }));
}


export class GraphEditor {
    private graph: RenderGraph | null = null;
    private pendingSource: string | null = null; // "Pass.field" of a clicked output port awaiting an input port
    private message = "";
    private readonly svg: SVGSVGElement;
    private readonly toolbar: HTMLDivElement;
    private readonly status: HTMLDivElement;
    private readonly properties: HTMLDivElement;
    /** Node positions set by dragging (override the layered layout). */
    private readonly positions = new Map<string, { x: number; y: number }>();
    private selected: string | null = null;
    private drag: { name: string; x0: number; y0: number; px: number; py: number; moved: boolean } | null = null;

    constructor(
        private readonly container: HTMLElement,
        private readonly hooks: GraphEditorHooks,
    ) {
        const doc = container.ownerDocument;
        this.toolbar = doc.createElement("div");
        this.toolbar.className = "graph-toolbar";
        const typeSel = doc.createElement("select");
        for (const t of getRegisteredRenderPasses().sort()) {
            const opt = doc.createElement("option");
            opt.value = t;
            opt.textContent = t;
            typeSel.appendChild(opt);
        }
        const nameInput = doc.createElement("input");
        nameInput.placeholder = "pass name";
        nameInput.size = 12;
        const addBtn = doc.createElement("button");
        addBtn.textContent = "Add pass";
        addBtn.onclick = () => {
            if (!this.graph) return;
            const type = typeSel.value;
            const name = nameInput.value.trim() || type;
            try {
                this.graph.addPass(createPass(this.graph.device, type), name);
                this.message = `added ${name} (${type})`;
                this.changed();
            } catch (e) {
                this.message = String(e);
                this.render();
            }
        };
        const saveBtn = doc.createElement("button");
        saveBtn.textContent = "Save graph .py";
        saveBtn.title = "Download the graph as an upstream-style python script (RenderGraphExporter)";
        saveBtn.onclick = () => {
            if (!this.graph) return;
            const a = doc.createElement("a");
            a.href = URL.createObjectURL(new Blob([this.graph.exportScript()], { type: "text/x-python" }));
            a.download = `${this.graph.name.replace(/\W+/g, "_") || "graph"}.py`;
            a.click();
            URL.revokeObjectURL(a.href);
        };
        this.toolbar.append(typeSel, nameInput, addBtn, saveBtn);
        const hint = doc.createElement("span");
        hint.className = "ui-text";
        hint.textContent = "drag a node to move it · click a node for its properties · click an output port then an input port to connect · click an edge to remove it · ★ marks a graph output · × removes a pass";
        this.toolbar.appendChild(hint);
        this.svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.svg.classList.add("graph-canvas");
        this.status = doc.createElement("div");
        this.status.className = "ui-text";
        this.properties = doc.createElement("div");
        this.properties.className = "graph-properties";
        container.append(this.toolbar, this.svg, this.status, this.properties);
        // Dragging moves a node; a press without movement selects it.
        const view = doc.defaultView ?? window;
        view.addEventListener("pointermove", (e) => {
            const d = this.drag;
            if (!d) return;
            const dx = e.clientX - d.x0;
            const dy = e.clientY - d.y0;
            if (!d.moved && Math.hypot(dx, dy) < 3) return;
            d.moved = true;
            this.positions.set(d.name, { x: Math.max(0, d.px + dx), y: Math.max(0, d.py + dy) });
            this.render();
        });
        view.addEventListener("pointerup", () => {
            const d = this.drag;
            this.drag = null;
            if (d && !d.moved) this.select(d.name);
        });
    }

    /** Selects a pass and shows its properties (its renderUI), or clears the selection. */
    select(name: string | null): void {
        this.selected = name;
        this.renderProperties();
        this.render();
    }

    private renderProperties(): void {
        const panel = this.properties;
        panel.innerHTML = "";
        const pass = this.selected && this.graph?.getPasses().find((p) => p.name === this.selected)?.pass;
        if (!pass) {
            this.selected = null;
            return;
        }
        const title = panel.ownerDocument.createElement("div");
        title.className = "graph-properties-title";
        title.textContent = `${this.selected} (${pass.type || pass.constructor.name})`;
        const body = panel.ownerDocument.createElement("div");
        pass.renderUI(new DomWidgets(body, () => this.hooks.onPassPropertiesChanged?.()));
        if (body.childElementCount === 0) body.textContent = "(no properties)";
        panel.append(title, body);
    }

    setGraph(graph: RenderGraph | null): void {
        if (graph !== this.graph) {
            this.positions.clear();
            this.selected = null;
        }
        this.graph = graph;
        this.pendingSource = null;
        this.renderProperties();
        this.render();
    }

    private changed(): void {
        this.pendingSource = null;
        this.hooks.onGraphChanged();
        this.renderProperties();
        this.render();
    }

    /** Layered layout: depth = longest incoming path; rows in insertion order within a column. */
    private layout(passes: { name: string; pass: RenderPass }[], edges: Edge[]): Map<string, { x: number; y: number; h: number; inputs: string[]; outputs: string[] }> {
        const depth = new Map<string, number>();
        const incoming = (n: string) => edges.filter((e) => e.dstPass === n);
        const visit = (n: string, guard: Set<string>): number => {
            if (depth.has(n)) return depth.get(n)!;
            if (guard.has(n)) return 0; // cycle guard
            guard.add(n);
            const d = incoming(n).reduce((m, e) => Math.max(m, visit(e.srcPass, guard) + 1), 0);
            depth.set(n, d);
            return d;
        };
        for (const { name } of passes) visit(name, new Set());
        const [w, h] = this.hooks.defaultTexDims();
        const columns = new Map<number, number>(); // depth -> next y
        const nodes = new Map<string, { x: number; y: number; h: number; inputs: string[]; outputs: string[] }>();
        for (const { name, pass } of passes) {
            const fields = pass.reflect({ defaultTexDims: [w, h] }).fields;
            const inputs = fields.filter((f) => f.isInput()).map((f) => f.name_);
            const outputs = fields.filter((f) => f.isOutput()).map((f) => f.name_);
            const d = depth.get(name) ?? 0;
            const y = columns.get(d) ?? 8;
            const height = 22 + Math.max(inputs.length, outputs.length) * kPortH + 6;
            const placed = this.positions.get(name);
            nodes.set(name, { x: placed?.x ?? 8 + d * (kNodeW + kColGap), y: placed?.y ?? y, h: height, inputs, outputs });
            columns.set(d, y + height + kRowGap);
        }
        return nodes;
    }

    render(): void {
        const svg = this.svg;
        while (svg.firstChild) svg.removeChild(svg.firstChild);
        this.status.textContent = this.message;
        if (!this.graph) return;
        const graph = this.graph;
        const passes = graph.getPasses();
        const edges = graphEdges(graph);
        const outputs = new Set(graph.getOutputNames());
        const nodes = this.layout(passes, edges);
        const ns = "http://www.w3.org/2000/svg";
        const el = <K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, text?: string): SVGElementTagNameMap[K] => {
            const e = svg.ownerDocument.createElementNS(ns, tag);
            for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
            if (text !== undefined) e.textContent = text;
            return e;
        };
        let maxX = 0;
        let maxY = 0;
        const portPos = (pass: string, field: string, output: boolean): [number, number] | null => {
            const n = nodes.get(pass);
            if (!n) return null;
            const list = output ? n.outputs : n.inputs;
            const i = list.indexOf(field);
            if (i < 0) return null;
            return [n.x + (output ? kNodeW : 0), n.y + 22 + i * kPortH + kPortH / 2];
        };
        // Edges first (under the nodes).
        for (const e of edges) {
            const a = portPos(e.srcPass, e.srcField, true);
            const b = portPos(e.dstPass, e.dstField, false);
            if (!a || !b) continue;
            const dx = Math.max(30, (b[0] - a[0]) / 2);
            const path = el("path", { d: `M${a[0]},${a[1]} C${a[0] + dx},${a[1]} ${b[0] - dx},${b[1]} ${b[0]},${b[1]}`, class: "graph-edge", "data-edge": `${e.srcPass}.${e.srcField}->${e.dstPass}.${e.dstField}` });
            path.addEventListener("click", () => {
                graph.removeEdge(`${e.srcPass}.${e.srcField}`, `${e.dstPass}.${e.dstField}`);
                this.message = `removed edge ${e.srcPass}.${e.srcField} -> ${e.dstPass}.${e.dstField}`;
                this.changed();
            });
            svg.appendChild(path);
        }
        for (const { name, pass } of passes) {
            const n = nodes.get(name)!;
            const g = el("g", { class: `graph-node${this.selected === name ? " graph-node-selected" : ""}`, "data-pass": name, transform: `translate(${n.x},${n.y})` });
            const bg = el("rect", { width: kNodeW, height: n.h, rx: 4, class: "graph-node-bg" });
            const title = el("text", { x: 6, y: 14, class: "graph-node-title" }, `${name} (${pass.type || pass.constructor.name})`);
            for (const handle of [bg, title]) {
                handle.addEventListener("pointerdown", (ev) => {
                    const e = ev as PointerEvent;
                    e.preventDefault();
                    this.drag = { name, x0: e.clientX, y0: e.clientY, px: n.x, py: n.y, moved: false };
                });
            }
            g.append(bg, title);
            const remove = el("text", { x: kNodeW - 12, y: 14, class: "graph-remove", "data-remove": name }, "×");
            remove.addEventListener("click", () => {
                graph.removePass(name);
                this.positions.delete(name);
                if (this.selected === name) this.selected = null;
                this.message = `removed ${name}`;
                this.changed();
            });
            g.appendChild(remove);
            n.inputs.forEach((f, i) => {
                const y = 22 + i * kPortH + kPortH / 2;
                const port = el("circle", { cx: 0, cy: y, r: 4, class: "graph-port graph-port-in", "data-port": `${name}.${f}` });
                port.addEventListener("click", () => {
                    if (!this.pendingSource) {
                        this.message = "select an output port first";
                        this.render();
                        return;
                    }
                    try {
                        graph.addEdge(this.pendingSource, `${name}.${f}`);
                        this.message = `connected ${this.pendingSource} -> ${name}.${f}`;
                        this.changed();
                    } catch (e) {
                        this.message = String(e);
                        this.pendingSource = null;
                        this.render();
                    }
                });
                g.append(port, el("text", { x: 8, y: y + 4, class: "graph-port-label" }, f));
            });
            n.outputs.forEach((f, i) => {
                const y = 22 + i * kPortH + kPortH / 2;
                const ref = `${name}.${f}`;
                const selected = this.pendingSource === ref;
                const port = el("circle", { cx: kNodeW, cy: y, r: 4, class: `graph-port graph-port-out${selected ? " graph-port-selected" : ""}`, "data-port": ref });
                port.addEventListener("click", () => {
                    this.pendingSource = selected ? null : ref;
                    this.message = selected ? "" : `source ${ref}: now click an input port`;
                    this.render();
                });
                const star = el("text", { x: kNodeW - 14, y: y + 4, class: `graph-mark${outputs.has(ref) ? " graph-marked" : ""}`, "data-mark": ref }, "★");
                star.addEventListener("click", () => {
                    if (outputs.has(ref)) graph.unmarkOutput(ref);
                    else graph.markOutput(ref);
                    this.message = `${outputs.has(ref) ? "unmarked" : "marked"} ${ref}`;
                    this.changed();
                });
                g.append(port, el("text", { x: kNodeW - 22, y: y + 4, class: "graph-port-label graph-port-label-out", "text-anchor": "end" }, f), star);
            });
            svg.appendChild(g);
            maxX = Math.max(maxX, n.x + kNodeW + 12);
            maxY = Math.max(maxY, n.y + n.h + 12);
        }
        svg.setAttribute("width", String(Math.max(320, maxX)));
        svg.setAttribute("height", String(Math.max(80, maxY)));
    }
}
