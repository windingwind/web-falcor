/**
 * EdgeAwareFilter: edge-preserving smoothing and detail manipulation in one pass, with three methods.
 *
 *  - `guided`        guided filter (He, Sun, Tang, ECCV 2010): grey guide, per-channel colour; 4 box-filter dispatches.
 *  - `bilateralGrid` bilateral grid (Chen, Paris, Durand, SIGGRAPH 2007): u32 fixed-point atomic splat, separable grid
 *                    blur, trilinear slice; 6 dispatches.
 *  - `localLaplacian` fast local Laplacian filter (Paris, Hasinoff, Kautz, SIGGRAPH 2011; Aubry et al., TOG 2014) on the
 *                    encoded luminance; about 2 + (2 + 2 * pyramidLevels) * intensitySamples dispatches.
 * Written from the papers; no code was taken from other implementations.
 *
 * `detail` < 0 removes detail (-1 = the edge-preserving base layer: abstraction), > 0 exaggerates it (+1 = twice the
 * detail). Guided filter and bilateral grid: result = src + detail * (src - base). Local Laplacian: remap
 * r(i) = i + detail * d * exp(-d^2 / (2 sigmaR^2)), d = i - reference intensity. Output = src + strength * (result - src),
 * clamped to >= 0; alpha is passed through. detail = 0 or strength = 0 copies src bit-exactly (no filtering runs).
 *
 * Guide value g = guideScale * guide.x if the `guide` input is connected (e.g. linear depth), else luminance^(1/2.2).
 * The bilateral grid clamps g to [0, 1]; the local Laplacian always works on luminance^(1/2.2) (it ignores `guide`).
 * The bilateral grid splats colour clamped to [0, 64] (fixed point, 1/1024 steps).
 *
 * Inputs : src, guide (optional). Output: dst (RGBA32Float, size of src).
 * Params : method, radius (guided, px), eps (guided), cellSize (grid, px), sigmaR (grid and local Laplacian range),
 *          pyramidLevels, intensitySamples (local Laplacian), detail, strength, guideScale.
 * The pass is stateless and keeps no history.
 */

import {
    Buffer,
    ComputePass,
    FieldFlags,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type PropertyValue,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/EdgeAwareFilter/EdgeAwareFilter.cs.slang";
export const kEdgeAwareMethods = ["guided", "bilateralGrid", "localLaplacian"] as const;
const kEntries = ["copyMain", "gfH1", "gfV1", "gfH2", "gfV2", "bgClear", "bgSplat", "bgBlurU", "bgBlur", "bgSlice",
    "llGuide", "llDown", "llRemap", "llLap", "llCollapse", "llFinal"] as const;
type Entry = (typeof kEntries)[number];
/** Largest bilateral grid (cells); 16 bytes per cell stays below WebGPU's default 128 MiB storage binding limit. */
const kMaxGridCells = 8_000_000;

const kDefaults = {
    method: "guided" as string,
    radius: 8,
    eps: 0.01,
    cellSize: 16,
    sigmaR: 0.15,
    pyramidLevels: 5,
    intensitySamples: 8,
    detail: -1,
    strength: 1,
    guideScale: 1,
};
type Params = typeof kDefaults;

const kRanges: Record<Exclude<keyof Params, "method">, [number, number, boolean]> = {
    radius: [1, 32, true],
    eps: [1e-6, 10, false],
    cellSize: [2, 64, true],
    sigmaR: [0.01, 1, false],
    pyramidLevels: [1, 10, true],
    intensitySamples: [2, 32, true],
    detail: [-1, 4, false],
    strength: [0, 1, false],
    guideScale: [-1e6, 1e6, false],
};

export class EdgeAwareFilter extends RenderPass {
    private p: Params = { ...kDefaults };
    private passes = new Map<Entry, ComputePass>();
    private tex = new Map<string, Texture>();
    private bufs = new Map<string, Buffer>();
    private dummy: Texture | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [k] of props.entries()) {
            if (!(k in kDefaults)) throw new Error(`EdgeAwareFilter: unknown property '${k}' (known: ${Object.keys(kDefaults).join(", ")})`);
        }
        this.p.method = String(props.get("method", kDefaults.method));
        if (!(kEdgeAwareMethods as readonly string[]).includes(this.p.method)) {
            throw new Error(`EdgeAwareFilter: method must be one of ${kEdgeAwareMethods.join("|")} (got ${this.p.method})`);
        }
        for (const [k, [lo, hi, int]] of Object.entries(kRanges) as [keyof typeof kRanges, [number, number, boolean]][]) {
            const v = Number(props.get(k, kDefaults[k]));
            if (!(v >= lo && v <= hi) || (int && !Number.isInteger(v))) {
                throw new Error(`EdgeAwareFilter: ${k} must be ${int ? "an integer " : ""}in [${lo}, ${hi}] (got ${String(props.get(k, kDefaults[k]))})`);
            }
            this.p[k] = v;
        }
    }

    override getProperties(): Properties {
        return new Properties({ ...this.p } as Record<string, PropertyValue>);
    }

    override renderUI(ui: UIWidgets): void {
        ui.text(`method: ${this.p.method}`);
        ui.slider("Detail", this.p.detail, -1, 4, 0.01, (v) => (this.p.detail = v));
        ui.slider("Strength", this.p.strength, 0, 1, 0.01, (v) => (this.p.strength = v));
        ui.slider("Range sigma", this.p.sigmaR, 0.01, 1, 0.01, (v) => (this.p.sigmaR = v));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("src", "Colour").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("guide", "Optional guide image (x channel, times guideScale), e.g. linear depth").bindFlags(ResourceBindFlags.ShaderResource).flags(FieldFlags.Optional);
        r.addOutput("dst", "Filtered colour").texture2D(w, h).format(ResourceFormat.RGBA32Float)
            .bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
        return r;
    }

    private pass(e: Entry): ComputePass {
        let p = this.passes.get(e);
        if (!p) this.passes.set(e, (p = ComputePass.create(this.device, { path: kShaderFile, csEntry: e })));
        return p;
    }

    private texture(key: string, w: number, h: number, format: ResourceFormat): Texture {
        let t = this.tex.get(key);
        if (!t || t.width !== w || t.height !== h) {
            t = this.device.createTexture2D(w, h, format, 1, 1, undefined, ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess);
            this.tex.set(key, t);
        }
        return t;
    }

    private buffer(key: string, elements: number, structSize: number): Buffer {
        let b = this.bufs.get(key);
        if (!b || b.size < elements * structSize) {
            b = new Buffer(this.device, {
                size: Math.max(elements, 1) * structSize, structSize,
                bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
                memoryType: MemoryType.DeviceLocal, name: `EdgeAwareFilter::${key}`,
            });
            this.bufs.set(key, b);
        }
        return b;
    }

    /** Binds the common constants plus `extra` (constants by CB name, resources by shader name) and dispatches. */
    private run(ctx: RenderContext, e: Entry, src: Texture, guide: Texture | null, extra: Record<string, unknown>, threads: [number, number?]): void {
        const pass = this.pass(e);
        const root = pass.getRootVar();
        const cb = root["CB"]!;
        cb["gFrameDim"] = [src.width, src.height];
        cb["gRadius"] = this.p.radius;
        cb["gEps"] = this.p.eps;
        cb["gDetail"] = this.p.detail;
        cb["gStrength"] = this.p.strength;
        cb["gHasGuide"] = guide ? 1 : 0;
        cb["gGuideScale"] = this.p.guideScale;
        cb["gInvSigmaR"] = 1 / this.p.sigmaR;
        root["gSrc"] = src;
        root["gGuide"] = guide ?? (this.dummy ??= this.device.createTexture2D(1, 1, ResourceFormat.RGBA32Float, 1, 1, new Float32Array([0, 0, 0, 1]), ResourceBindFlags.ShaderResource));
        for (const [k, v] of Object.entries(extra)) {
            if (v instanceof Texture || v instanceof Buffer) root[k] = v;
            else cb[k] = v as number;
        }
        pass.execute(ctx, threads[0], threads[1] ?? 1);
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const src = rd.getTexture("src"), dst = rd.getTexture("dst");
        if (!src || !dst) return;
        const guide = rd.getTexture("guide") ?? null;
        const W = dst.width, H = dst.height;
        const full: [number, number] = [W, H];
        if (this.p.detail === 0 || this.p.strength === 0) {
            this.run(ctx, "copyMain", src, guide, { gDst: dst }, full);
            return;
        }
        if (this.p.method === "guided") this.guided(ctx, src, guide, dst);
        else if (this.p.method === "bilateralGrid") this.bilateralGrid(ctx, src, guide, dst);
        else this.localLaplacian(ctx, src, dst);
    }

    private guided(ctx: RenderContext, src: Texture, guide: Texture | null, dst: Texture): void {
        const W = dst.width, H = dst.height, f = ResourceFormat.RGBA32Float;
        const [a1, b1, a2, b2] = ["a1", "b1", "a2", "b2"].map((k) => this.texture(k, W, H, f)) as [Texture, Texture, Texture, Texture];
        this.run(ctx, "gfH1", src, guide, { gOutA: a1, gOutB: b1 }, [W, H]);
        this.run(ctx, "gfV1", src, guide, { gInA: a1, gInB: b1, gOutA: a2, gOutB: b2 }, [W, H]);
        this.run(ctx, "gfH2", src, guide, { gInA: a2, gInB: b2, gOutA: a1, gOutB: b1 }, [W, H]);
        this.run(ctx, "gfV2", src, guide, { gInA: a1, gInB: b1, gDst: dst }, [W, H]);
    }

    /** Grid size for a frame: (ceil(W / cellSize), ceil(H / cellSize), round(1 / sigmaR) + 1). */
    static gridDims(W: number, H: number, cellSize: number, sigmaR: number): [number, number, number] {
        return [Math.ceil(W / cellSize), Math.ceil(H / cellSize), Math.round(1 / sigmaR) + 1];
    }

    private bilateralGrid(ctx: RenderContext, src: Texture, guide: Texture | null, dst: Texture): void {
        const W = dst.width, H = dst.height;
        const [gw, gh, gd] = EdgeAwareFilter.gridDims(W, H, this.p.cellSize, this.p.sigmaR);
        const n = gw * gh * gd;
        if (n > kMaxGridCells) throw new Error(`EdgeAwareFilter: bilateral grid of ${gw}x${gh}x${gd} cells is too large (max ${kMaxGridCells}); raise cellSize or sigmaR`);
        const u = this.buffer("gridU", 4 * n, 4), g0 = this.buffer("grid0", n, 16), g1 = this.buffer("grid1", n, 16);
        const grid = { gGridW: gw, gGridH: gh, gGridD: gd, gCellSize: this.p.cellSize };
        this.run(ctx, "bgClear", src, guide, { ...grid, gGridU: u }, [4 * n]);
        this.run(ctx, "bgSplat", src, guide, { ...grid, gGridU: u }, [W, H]);
        this.run(ctx, "bgBlurU", src, guide, { ...grid, gAxis: 0, gGridU: u, gGridOut: g0 }, [n]);
        this.run(ctx, "bgBlur", src, guide, { ...grid, gAxis: 1, gGridIn: g0, gGridOut: g1 }, [n]);
        this.run(ctx, "bgBlur", src, guide, { ...grid, gAxis: 2, gGridIn: g1, gGridOut: g0 }, [n]);
        this.run(ctx, "bgSlice", src, guide, { ...grid, gGridIn: g0, gDst: dst }, [W, H]);
    }

    /** Level sizes of the pyramid: level 0 = W x H, each next level ceil(n / 2); levels are capped so the top is >= 2 px. */
    static pyramidDims(W: number, H: number, levels: number): [number, number][] {
        const dims: [number, number][] = [[W, H]];
        while (dims.length <= levels) {
            const [w, h] = dims[dims.length - 1]!;
            if (w < 4 || h < 4) break;
            dims.push([Math.ceil(w / 2), Math.ceil(h / 2)]);
        }
        return dims;
    }

    private localLaplacian(ctx: RenderContext, src: Texture, dst: Texture): void {
        const W = dst.width, H = dst.height, f = ResourceFormat.R32Float;
        const dims = EdgeAwareFilter.pyramidDims(W, H, this.p.pyramidLevels);
        const n = dims.length - 1; // Laplacian levels 0..n-1, Gaussian levels 0..n
        if (n < 1) throw new Error(`EdgeAwareFilter: localLaplacian needs an image of at least 4x4 px (got ${W}x${H})`);
        const K = this.p.intensitySamples;
        const gin = dims.map(([w, h], l) => this.texture(`gin${l}`, w, h, f));
        const rk = dims.map(([w, h], l) => this.texture(`rk${l}`, w, h, f));
        const acc = dims.slice(0, n).map(([w, h], l) => this.buffer(`acc${l}`, w * h, 4));
        const lvl = (l: number) => ({ gOutDim: dims[l]!, gInDim: dims[l + 1] ?? dims[l]! });
        const down = (from: Texture[], l: number) =>
            this.run(ctx, "llDown", src, null, { gInDim: dims[l]!, gOutDim: dims[l + 1]!, gInL: from[l]!, gOutL: from[l + 1]! }, dims[l + 1]!);
        this.run(ctx, "llGuide", src, null, { gOutL: gin[0]! }, [W, H]);
        for (let l = 0; l < n; l++) down(gin, l);
        for (let k = 0; k < K; k++) {
            const ref = k / (K - 1);
            this.run(ctx, "llRemap", src, null, { gRef: ref, gInL: gin[0]!, gOutL: rk[0]! }, [W, H]);
            for (let l = 0; l < n; l++) down(rk, l);
            for (let l = 0; l < n; l++) {
                this.run(ctx, "llLap", src, null, { ...lvl(l), gRef: ref, gRefStep: K - 1, gFirst: k === 0 ? 1 : 0,
                    gInL: gin[l]!, gGuideL: rk[l]!, gCoarseL: rk[l + 1]!, gAcc: acc[l]! }, dims[l]!);
            }
        }
        // collapse: out_n = gin_n; out_l (l >= 1) written into rk_l; level 0 goes straight to dst
        for (let l = n - 1; l >= 1; l--) {
            this.run(ctx, "llCollapse", src, null, { ...lvl(l), gAcc: acc[l]!, gCoarseL: l === n - 1 ? gin[n]! : rk[l + 1]!, gOutL: rk[l]! }, dims[l]!);
        }
        this.run(ctx, "llFinal", src, null, { ...lvl(0), gAcc: acc[0]!, gCoarseL: n === 1 ? gin[1]! : rk[1]!, gDst: dst }, [W, H]);
    }
}

registerRenderPass("EdgeAwareFilter", (device, props) => new EdgeAwareFilter(device, props), {
    desc: "Edge-preserving smoothing / detail enhancement (guided filter, bilateral grid, fast local Laplacian)",
});
