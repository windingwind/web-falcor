/**
 * Falcor's Python UI (Utils/UI/PythonUI: `falcor.ui` Screen, Window, Group, Text,
 * ProgressBar, Button, Checkbox, Combobox, Drag and Slider widgets) as DOM elements.
 * §9: native renders these with ImGui inside Testbed::frame and runs callbacks there;
 * here user edits are queued and the Python side drains them in `testbed.frame()`,
 * so callbacks still run inside frame() (and never re-enter a suspended script).
 */

export type PyUiKind = "window" | "group" | "text" | "progress" | "button" | "checkbox" | "combobox" | "drag" | "slider";

interface PyUiWidget {
    kind: PyUiKind;
    el: HTMLElement;
    /** Where children go. */
    content: HTMLElement;
    props: Record<string, unknown>;
    visible: boolean;
    enabled: boolean;
    update(): void;
}

/** One Testbed's UI: an overlay layer holding its windows; widget ids are indices. */
export class PyUiScreen {
    readonly root: HTMLElement;
    private widgets: (PyUiWidget | null)[] = [];
    private events: [number, unknown][] = [];

    constructor(host: HTMLElement) {
        this.root = document.createElement("div");
        this.root.className = "falcor-ui-screen";
        Object.assign(this.root.style, { position: "absolute", left: "0", top: "0", width: "100%", height: "100%", pointerEvents: "none", font: "13px sans-serif", overflow: "hidden" });
        if (getComputedStyle(host).position === "static") host.style.position = "relative";
        host.appendChild(this.root);
        this.widgets.push({ kind: "window", el: this.root, content: this.root, props: {}, visible: true, enabled: true, update: () => {} });
    }

    /** Edits since the last call, as [widget id, new value] (buttons: null). */
    takeEvents(): [number, unknown][] {
        const e = this.events;
        this.events = [];
        return e;
    }

    create(kind: PyUiKind, parent: number, props: Record<string, unknown>): number {
        const id = this.widgets.length;
        const w = this.build(id, kind, props);
        this.widgets.push(w);
        this.setParent(id, parent);
        w.update();
        return id;
    }

    setParent(id: number, parent: number | null): void {
        const w = this.widgets[id];
        if (!w) return;
        w.el.remove();
        if (parent !== null) this.widgets[parent]?.content.appendChild(w.el);
    }

    set(id: number, name: string, value: unknown): void {
        const w = this.widgets[id];
        if (!w) return;
        if (name === "visible") w.visible = Boolean(value);
        else if (name === "enabled") w.enabled = Boolean(value);
        else w.props[name] = value;
        w.update();
    }

    private build(id: number, kind: PyUiKind, props: Record<string, unknown>): PyUiWidget {
        const el = document.createElement(kind === "group" ? "details" : "div");
        const w: PyUiWidget = { kind, el, content: el, props, visible: true, enabled: true, update: () => {} };
        const push = (value: unknown) => this.events.push([id, value]);
        const row = () => Object.assign(el.style, { margin: "3px 0", display: "flex", gap: "6px", alignItems: "center" });
        const labelSpan = () => {
            const s = document.createElement("span");
            s.style.whiteSpace = "nowrap";
            return s;
        };
        const show = () => (el.style.display = w.visible ? (kind === "group" || kind === "window" ? "block" : "flex") : "none");
        switch (kind) {
            case "window": {
                Object.assign(el.style, { position: "absolute", pointerEvents: "auto", background: "rgba(30,30,34,0.92)", color: "#ddd", border: "1px solid #555", borderRadius: "4px", boxSizing: "border-box", display: "flex", flexDirection: "column" });
                const bar = document.createElement("div");
                Object.assign(bar.style, { background: "#2d4a73", padding: "3px 6px", cursor: "move", display: "flex", justifyContent: "space-between", userSelect: "none" });
                const title = document.createElement("span");
                const close = document.createElement("span");
                close.textContent = "×";
                close.style.cursor = "pointer";
                close.onclick = () => {
                    w.visible = false;
                    w.update();
                };
                bar.append(title, close);
                const content = document.createElement("div");
                Object.assign(content.style, { padding: "6px", overflow: "auto", flex: "1" });
                el.append(bar, content);
                w.content = content;
                // Drag the title bar to move the window (ImGui-style).
                bar.onpointerdown = (e) => {
                    const [x0, y0] = [e.clientX, e.clientY];
                    const [px, py] = ((props.position as number[]) ?? [10, 10]) as [number, number];
                    const move = (m: PointerEvent) => {
                        props.position = [px + m.clientX - x0, py + m.clientY - y0];
                        w.update();
                    };
                    const up = () => {
                        window.removeEventListener("pointermove", move);
                        window.removeEventListener("pointerup", up);
                    };
                    window.addEventListener("pointermove", move);
                    window.addEventListener("pointerup", up);
                };
                w.update = () => {
                    const [x, y] = (props.position as number[]) ?? [10, 10];
                    const [sw, sh] = (props.size as number[]) ?? [400, 400];
                    Object.assign(el.style, { left: `${x}px`, top: `${y}px`, width: `${sw}px`, height: `${sh}px`, display: w.visible ? "flex" : "none" });
                    title.textContent = String(props.title ?? "");
                };
                break;
            }
            case "group": {
                const d = el as HTMLDetailsElement;
                d.open = true;
                Object.assign(d.style, { margin: "4px 0" });
                const summary = document.createElement("summary");
                Object.assign(summary.style, { background: "#34405a", padding: "2px 4px", cursor: "pointer" });
                const body = document.createElement("div");
                body.style.padding = "2px 0 2px 10px";
                d.append(summary, body);
                w.content = body;
                w.update = () => {
                    summary.textContent = String(props.label ?? "");
                    show();
                };
                break;
            }
            case "text":
                el.style.whiteSpace = "pre-wrap";
                w.update = () => {
                    el.textContent = String(props.text ?? "");
                    el.style.opacity = w.enabled ? "1" : "0.5";
                    el.style.display = w.visible ? "block" : "none";
                };
                break;
            case "progress": {
                row();
                const bar = document.createElement("progress");
                bar.max = 1;
                bar.style.flex = "1";
                const pct = labelSpan();
                el.append(bar, pct);
                w.update = () => {
                    const f = Math.min(Math.max(Number(props.fraction ?? 0), 0), 1);
                    bar.value = f;
                    pct.textContent = `${Math.round(f * 100)}%`;
                    show();
                };
                break;
            }
            case "button": {
                row();
                const b = document.createElement("button");
                b.onclick = () => push(null);
                el.append(b);
                w.update = () => {
                    b.textContent = String(props.label ?? "");
                    b.disabled = !w.enabled;
                    show();
                };
                break;
            }
            case "checkbox": {
                row();
                const input = document.createElement("input");
                input.type = "checkbox";
                const label = labelSpan();
                input.onchange = () => push(input.checked);
                el.append(input, label);
                w.update = () => {
                    input.checked = Boolean(props.value);
                    input.disabled = !w.enabled;
                    label.textContent = String(props.label ?? "");
                    show();
                };
                break;
            }
            case "combobox": {
                row();
                const select = document.createElement("select");
                const label = labelSpan();
                select.onchange = () => push(select.selectedIndex);
                el.append(select, label);
                w.update = () => {
                    const items = (props.items as string[]) ?? [];
                    if (select.options.length !== items.length || items.some((t, i) => select.options[i]!.text !== t)) {
                        select.replaceChildren(...items.map((t) => new Option(t)));
                    }
                    select.selectedIndex = Number(props.value ?? 0);
                    select.disabled = !w.enabled;
                    label.textContent = String(props.label ?? "");
                    show();
                };
                break;
            }
            case "drag":
            case "slider": {
                row();
                const n = Number(props.components ?? 1);
                const isInt = Boolean(props.integer);
                const inputs = Array.from({ length: n }, () => {
                    const input = document.createElement("input");
                    input.type = kind === "slider" ? "range" : "number";
                    input.style.width = kind === "slider" ? "90px" : "70px";
                    return input;
                });
                const readout = labelSpan();
                const label = labelSpan();
                const current = () => (Array.isArray(props.value) ? (props.value as number[]) : [Number(props.value ?? 0)]);
                inputs.forEach((input, k) => {
                    input.oninput = () => {
                        let v = Number(input.value);
                        if (isInt) v = Math.round(v);
                        const next = current().slice();
                        next[k] = v;
                        props.value = n === 1 ? next[0] : next;
                        push(props.value);
                        w.update();
                    };
                });
                el.append(...inputs, ...(kind === "slider" ? [readout] : []), label);
                w.update = () => {
                    const [min, max] = [Number(props.min ?? 0), Number(props.max ?? 0)];
                    const bounded = max > min;
                    const step = isInt ? 1 : kind === "drag" ? Number(props.speed ?? 1) * 0.01 || "any" : (max - min) / 1000 || "any";
                    inputs.forEach((input, k) => {
                        input.min = bounded ? String(min) : "";
                        input.max = bounded ? String(max) : "";
                        input.step = String(step);
                        if (document.activeElement !== input) input.value = String(current()[k] ?? 0);
                        input.disabled = !w.enabled;
                    });
                    readout.textContent = current().map((v) => formatValue(String(props.format ?? (isInt ? "%d" : "%.3f")), v)).join(", ");
                    label.textContent = String(props.label ?? "");
                    show();
                };
                break;
            }
        }
        return w;
    }
}

/** A printf-style `%d` / `%.Nf` format of one number (ImGui's slider formats). */
function formatValue(format: string, v: number): string {
    return format.replace(/%(\.(\d+))?([dfi])/, (_m, _p, digits, t) => (t === "f" ? v.toFixed(digits !== undefined ? Number(digits) : 6) : String(Math.round(v))));
}
