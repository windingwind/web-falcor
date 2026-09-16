/**
 * Profiler panel mirroring Utils/Timing/ProfilerUI.cpp on the DOM: options
 * (Pause / Average / Graph / capture), an Event | CPU | CPU % | GPU | GPU %
 * table indented by nesting level with percentage bars, and a stacked
 * history graph (256 frames) on a canvas with hover highlighting.
 */

import type { Profiler, ProfilerEvent, ProfilerStats } from "../../Core/API/Profiler.js";

export enum ProfilerGraphMode {
    Off,
    CpuTime,
    GpuTime,
}

const kGraphModes = ["Off", "CPU Time", "GPU Time"];
const kHistoryCapacity = 256;
const kGraphBarWidth = 2;
const kIndentWidth = 16;
// Colorblind-friendly palette (native kColorPalette).
const kColorPalette = ["#004949", "#009292", "#ff6db6", "#ffb6db", "#490092", "#006ddb", "#b66dff", "#6db6ff", "#b6dbff", "#920000", "#24ff24", "#ffff6d"];
const kHighlightColor = "rgba(255,127,0,0.81)";

interface EventData {
    event: ProfilerEvent;
    name: string;
    level: number;
    color: string;
    cpuTime: number;
    gpuTime: number;
    graphValue: number;
    maxGraphValue: number;
    graphHistory: Float32Array;
    row: HTMLTableRowElement;
}

function fmtStats(name: string, s: ProfilerStats): string {
    return `${name}\nMin: ${s.min.toFixed(2)}\nMax: ${s.max.toFixed(2)}\nMean: ${s.mean.toFixed(2)}\nStdDev: ${s.stdDev.toFixed(2)}`;
}

export class ProfilerUI {
    graphMode = ProfilerGraphMode.Off;
    enableAverage = true;
    private eventData: EventData[] = [];
    private totalCpuTime = 0;
    private totalGpuTime = 0;
    private historyWrite = 0;
    private historyLength = 0;
    private highlightIndex = -1;
    private readonly table: HTMLTableElement;
    private readonly tbody: HTMLTableSectionElement;
    private readonly graphCanvas: HTMLCanvasElement;
    private readonly graphTooltip: HTMLDivElement;
    private readonly captureButton: HTMLButtonElement;
    private mouse: [number, number] | null = null;

    constructor(
        private readonly profiler: Profiler,
        private readonly container: HTMLElement,
    ) {
        container.classList.add("profiler-ui");
        const doc = container.ownerDocument;
        const options = doc.createElement("div");
        options.className = "profiler-options";
        const check = (label: string, value: boolean, onChange: (v: boolean) => void) => {
            const l = doc.createElement("label");
            const input = doc.createElement("input");
            input.type = "checkbox";
            input.checked = value;
            input.addEventListener("change", () => onChange(input.checked));
            l.append(input, ` ${label}`);
            options.appendChild(l);
            return input;
        };
        check("Pause", profiler.isPaused(), (v) => profiler.setPaused(v));
        check("Average", this.enableAverage, (v) => (this.enableAverage = v));
        const graphLabel = doc.createElement("label");
        const graphSelect = doc.createElement("select");
        kGraphModes.forEach((m, i) => {
            const opt = doc.createElement("option");
            opt.value = String(i);
            opt.textContent = m;
            graphSelect.appendChild(opt);
        });
        graphSelect.addEventListener("change", () => {
            this.graphMode = Number(graphSelect.value) as ProfilerGraphMode;
            this.clearGraphData();
        });
        graphLabel.append("Graph ", graphSelect);
        options.appendChild(graphLabel);
        this.captureButton = doc.createElement("button");
        this.captureButton.textContent = "Start Capture";
        this.captureButton.addEventListener("click", () => {
            if (profiler.isCapturing()) {
                const capture = profiler.endCapture();
                if (capture) downloadText(doc, "profiler_capture.json", capture.toJsonString());
            } else {
                profiler.startCapture();
            }
        });
        options.appendChild(this.captureButton);
        container.appendChild(options);

        this.table = doc.createElement("table");
        const head = doc.createElement("thead");
        head.innerHTML = "<tr><th>Event</th><th>CPU Time</th><th>CPU %</th><th>GPU Time</th><th>GPU %</th></tr>";
        this.tbody = doc.createElement("tbody");
        this.table.append(head, this.tbody);
        container.appendChild(this.table);

        this.graphCanvas = doc.createElement("canvas");
        this.graphCanvas.className = "profiler-graph";
        this.graphCanvas.hidden = true;
        this.graphCanvas.addEventListener("mousemove", (e) => {
            const r = this.graphCanvas.getBoundingClientRect();
            this.mouse = [e.clientX - r.left, e.clientY - r.top];
        });
        this.graphCanvas.addEventListener("mouseleave", () => (this.mouse = null));
        this.graphTooltip = doc.createElement("div");
        this.graphTooltip.className = "profiler-tooltip";
        this.graphTooltip.hidden = true;
        container.append(this.graphCanvas, this.graphTooltip);
    }

    /** Mirrors ProfilerUI::render (call once per frame while visible). */
    render(): void {
        this.updateEventData();
        this.updateGraphData();
        this.captureButton.textContent = this.profiler.isCapturing() ? "End Capture" : "Start Capture";
        for (let i = 0; i < this.eventData.length; i++) {
            const d = this.eventData[i]!;
            const cells = d.row.cells;
            cells[0]!.style.paddingLeft = `${d.level * kIndentWidth}px`;
            cells[0]!.style.color = i === this.highlightIndex ? kHighlightColor : "";
            cells[1]!.textContent = `${d.cpuTime.toFixed(2)} ms`;
            cells[3]!.textContent = `${d.gpuTime.toFixed(2)} ms`;
            const cpuFrac = this.totalCpuTime > 0 ? d.cpuTime / this.totalCpuTime : 0;
            const gpuFrac = this.totalGpuTime > 0 ? d.gpuTime / this.totalGpuTime : 0;
            this.setBar(cells[2]!, cpuFrac, this.graphMode === ProfilerGraphMode.CpuTime ? d.color : "#fff", i === this.highlightIndex && this.graphMode === ProfilerGraphMode.CpuTime);
            this.setBar(cells[4]!, gpuFrac, this.graphMode === ProfilerGraphMode.GpuTime ? d.color : "#fff", i === this.highlightIndex && this.graphMode === ProfilerGraphMode.GpuTime);
            cells[2]!.title = `${d.name}\n${(cpuFrac * 100).toFixed(1)}%`;
            cells[4]!.title = `${d.name}\n${(gpuFrac * 100).toFixed(1)}%`;
        }
        const showGraph = this.graphMode !== ProfilerGraphMode.Off;
        this.graphCanvas.hidden = !showGraph;
        if (showGraph) this.renderGraph();
        else this.graphTooltip.hidden = true;
    }

    private setBar(cell: HTMLTableCellElement, fraction: number, color: string, highlight: boolean): void {
        const bar = cell.firstElementChild as HTMLDivElement;
        bar.style.width = `${Math.min(1, Math.max(0, fraction)) * 100}%`;
        bar.style.background = color;
        cell.style.outline = highlight ? `1px solid ${kHighlightColor}` : "";
    }

    /** Mirrors ProfilerUI::updateEventData (rows are rebuilt when the event set changes). */
    private updateEventData(): void {
        const events = this.profiler.getEvents();
        const same = events.length === this.eventData.length && events.every((e, i) => this.eventData[i]!.event === e);
        if (!same) {
            this.tbody.textContent = "";
            this.eventData = events.map((event, i) => {
                const row = this.tbody.insertRow();
                const nameCell = row.insertCell();
                nameCell.textContent = event.shortName;
                const cpuCell = row.insertCell();
                cpuCell.addEventListener("mouseenter", () => (cpuCell.title = fmtStats(event.shortName, event.computeCpuTimeStats())));
                this.barCell(row.insertCell(), i);
                const gpuCell = row.insertCell();
                gpuCell.addEventListener("mouseenter", () => (gpuCell.title = fmtStats(event.shortName, event.computeGpuTimeStats())));
                this.barCell(row.insertCell(), i);
                const prev = this.eventData.find((d) => d.event === event);
                return {
                    event,
                    name: event.shortName,
                    level: event.level,
                    color: kColorPalette[i % kColorPalette.length]!,
                    cpuTime: 0,
                    gpuTime: 0,
                    graphValue: 0,
                    maxGraphValue: prev?.maxGraphValue ?? 0,
                    graphHistory: prev?.graphHistory ?? new Float32Array(kHistoryCapacity),
                    row,
                };
            });
        }
        this.totalCpuTime = 0;
        this.totalGpuTime = 0;
        for (const d of this.eventData) {
            d.cpuTime = this.enableAverage ? Math.max(d.event.cpuTimeAverage, 0) : d.event.cpuTime;
            d.gpuTime = this.enableAverage ? Math.max(d.event.gpuTimeAverage, 0) : d.event.gpuTime;
            if (d.level === 0) {
                this.totalCpuTime += d.cpuTime;
                this.totalGpuTime += d.gpuTime;
            }
        }
    }

    private barCell(cell: HTMLTableCellElement, index: number): void {
        cell.className = "profiler-bar";
        const bar = cell.ownerDocument.createElement("div");
        cell.appendChild(bar);
        cell.addEventListener("mouseenter", () => {
            if (this.graphMode !== ProfilerGraphMode.Off) this.highlightIndex = index;
        });
    }

    /** Mirrors ProfilerUI::updateGraphData. */
    private updateGraphData(): void {
        if (this.graphMode === ProfilerGraphMode.Off) return;
        for (const d of this.eventData) {
            d.graphValue = this.graphMode === ProfilerGraphMode.CpuTime ? d.cpuTime : d.gpuTime;
            d.graphHistory[this.historyWrite] = d.graphValue;
            let max = 0;
            for (let j = 0; j < this.historyLength; j++) max = Math.max(max, d.graphHistory[j]!);
            d.maxGraphValue = max;
        }
        if (!this.profiler.isPaused()) {
            this.historyWrite = (this.historyWrite + 1) % kHistoryCapacity;
            this.historyLength = Math.min(this.historyLength + 1, kHistoryCapacity);
        }
    }

    private clearGraphData(): void {
        this.historyLength = 0;
        this.historyWrite = 0;
        for (const d of this.eventData) {
            d.graphValue = 0;
            d.maxGraphValue = 0;
        }
    }

    /** Mirrors ProfilerUI::renderGraph: newest frame at the left, level-0 stack with children nested. */
    private renderGraph(): void {
        const canvas = this.graphCanvas;
        const width = Math.max(1, this.container.clientWidth - 16);
        const height = Math.max(40, this.eventData.length * 18);
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, width, height);
        let totalMax = 0;
        for (const d of this.eventData) if (d.level === 0) totalMax += d.maxGraphValue;
        if (totalMax <= 0) return;
        const scaleY = height / totalMax;
        const levelY: number[] = new Array(128).fill(0);
        let newHighlight = -1;
        let highlightValue: number | null = null;
        let x = 0;
        for (let k = 0; k < this.historyLength && x <= width; k++) {
            const hi = (this.historyWrite + kHistoryCapacity - k - 1) % kHistoryCapacity;
            let totalValue = 0;
            for (const d of this.eventData) if (d.level === 0) totalValue += d.graphHistory[hi]!;
            levelY[0] = height - totalValue * scaleY;
            let hlY = 0;
            let hlH = 0;
            this.eventData.forEach((d, i) => {
                const value = d.graphHistory[hi]!;
                const y = levelY[d.level]!;
                const h = value * scaleY;
                ctx.fillStyle = d.color;
                ctx.fillRect(x, y, kGraphBarWidth, h);
                if (this.mouse && this.mouse[0] >= x && this.mouse[0] < x + kGraphBarWidth && this.mouse[1] >= y && this.mouse[1] < y + h) {
                    newHighlight = i;
                    highlightValue = totalValue > 0 ? value / totalValue : 0;
                }
                if (this.highlightIndex === i) {
                    hlY = y;
                    hlH = h;
                }
                levelY[d.level + 1] = levelY[d.level]!;
                levelY[d.level] = y + h;
            });
            if (hlH > 0) {
                ctx.fillStyle = kHighlightColor;
                ctx.fillRect(x, hlY, kGraphBarWidth, hlH);
            }
            x += kGraphBarWidth;
        }
        if (newHighlight >= 0) this.highlightIndex = newHighlight;
        if (this.mouse && highlightValue !== null && newHighlight >= 0) {
            this.graphTooltip.hidden = false;
            this.graphTooltip.textContent = `${this.eventData[newHighlight]!.name} ${((highlightValue as number) * 100).toFixed(2)}%`;
            this.graphTooltip.style.left = `${this.mouse[0] + 12}px`;
            this.graphTooltip.style.top = `${canvas.offsetTop + this.mouse[1] + 12}px`;
        } else {
            this.graphTooltip.hidden = true;
        }
    }
}

function downloadText(doc: Document, filename: string, text: string): void {
    const a = doc.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type: "application/json" }));
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}
