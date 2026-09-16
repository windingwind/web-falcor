// Render-graph editor panel (DOM/SVG analog of Falcor's RenderGraphEditor): passes laid out by
// dependency depth with input/output ports, edges as curves, and the graph's editing API
// (addPass/removePass, addEdge/removeEdge, markOutput/unmarkOutput) wired to clicks.
import { createPass, getRegisteredRenderPasses, type RenderGraph, type RenderPass } from "@web-falcor/falcor";

export interface GraphEditorHooks {
    /** Called after any edit (viewer: refresh outputs, rebuild pass panels, restart accumulation). */
    onGraphChanged: () => void;
    /** Graph output dimensions (reflection needs them). */
    defaultTexDims: () => [number, number];
}

interface Edge {
    srcPass: string;
    srcField: string;
    dstPass: string;
    dstField: string;
}

const kNodeW = 170;
const kPortH = 16;
const kColGap = 60;
const kRowGap = 24;

/** Graph edges: RenderGraph.getEdges() when available, else parsed from the script export (`g.addEdge("A.x", "B.y")` lines). */
export function graphEdges(graph: RenderGraph): Edge[] {
    const getEdges = (graph as unknown as { getEdges?: () => readonly Edge[] }).getEdges;
    if (typeof getEdges === "function") return getEdges.call(graph).map((e) => ({ ...e }));
    const edges: Edge[] = [];
    const re = /g\.addEdge\("([^"]+)", "([^"]+)"\)/g;
    for (const m of graph.exportScript().matchAll(re)) {
        const [srcPass, srcField] = splitRef(m[1]!);
        const [dstPass, dstField] = splitRef(m[2]!);
        edges.push({ srcPass, srcField, dstPass, dstField });
    }
    return edges;
}

function splitRef(ref: string): [string, string] {
    const i = ref.lastIndexOf(".");
    return [ref.slice(0, i), ref.slice(i + 1)];
}

export class GraphEditor {
    private graph: RenderGraph | null = null;
    private pendingSource: string | null = null; // "Pass.field" of a clicked output port awaiting an input port
    private message = "";
    private readonly svg: SVGSVGElement;
    private readonly toolbar: HTMLDivElement;
    private readonly status: HTMLDivElement;

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
        this.toolbar.append(typeSel, nameInput, addBtn);
        const hint = doc.createElement("span");
        hint.className = "ui-text";
        hint.textContent = "click an output port then an input port to connect · click an edge to remove it · ★ marks a graph output · × removes a pass";
        this.toolbar.appendChild(hint);
        this.svg = doc.createElementNS("http://www.w3.org/2000/svg", "svg");
        this.svg.classList.add("graph-canvas");
        this.status = doc.createElement("div");
        this.status.className = "ui-text";
        container.append(this.toolbar, this.svg, this.status);
    }

    setGraph(graph: RenderGraph | null): void {
        this.graph = graph;
        this.pendingSource = null;
        this.render();
    }

    private changed(): void {
        this.pendingSource = null;
        this.hooks.onGraphChanged();
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
            nodes.set(name, { x: 8 + d * (kNodeW + kColGap), y, h: height, inputs, outputs });
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
            const g = el("g", { class: "graph-node", "data-pass": name, transform: `translate(${n.x},${n.y})` });
            g.appendChild(el("rect", { width: kNodeW, height: n.h, rx: 4, class: "graph-node-bg" }));
            g.appendChild(el("text", { x: 6, y: 14, class: "graph-node-title" }, `${name} (${pass.type || pass.constructor.name})`));
            const remove = el("text", { x: kNodeW - 12, y: 14, class: "graph-remove", "data-remove": name }, "×");
            remove.addEventListener("click", () => {
                graph.removePass(name);
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
