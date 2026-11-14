/**
 * History helpers for passes that keep state from one frame to the next (temporal filters, feedback effects, particle
 * simulations, reflection/GI accumulation).
 *
 * The rule they implement: a pass's output must not depend on what was rendered before the current shot. A pass that
 * renders frame X on a fresh context and a pass that renders A, B, then X after a camera cut must produce bit-identical
 * output for X. Two things leak otherwise: stale history textures and the camera's previous-frame matrix (motion
 * vectors on the first frame after a cut).
 *
 * HistoryTracker decides once per execute whether the history is still valid and counts frames since the last reset.
 * It resets on:
 *  - the first execute of the pass instance (a fresh context or graph);
 *  - an explicit reset() (passes expose it as resetHistory(); resetGraphHistory(graph) calls it on every pass);
 *  - a different scene or a different active camera;
 *  - a change of the caller's `key` (e.g. resolution or format; the caller decides what invalidates its state);
 *  - Camera.resetHistory() (the camera's historyEpoch changed);
 *  - a primed cut: the camera has no motion this frame (previous = current view-projection) but differs from the camera
 *    this pass saw on its last frame, i.e. the camera was moved and re-primed by another render in between.
 * A camera that jumps without either of the last two looks like fast motion and keeps the history; signal cuts with
 * Camera.resetHistory() (or reset the passes) when rendering independent shots on one context.
 *
 * PingPongTextures is a pair of same-size textures alternated by frame index (write = [frameIndex & 1], read = the
 * other). On every reset and reallocation both are cleared to zero, so the state after a reset is the state of a newly
 * created pair: nothing from earlier frames can be read back.
 *
 * Usage in a pass:
 *   private history = new HistoryTracker();
 *   private pp = new PingPongTextures(device, ResourceFormat.RGBA32Float, "MyPass::history");
 *   resetHistory() { this.history.reset(); }
 *   execute(ctx, rd) {
 *       const f = this.history.begin(this.scene, `${w}x${h}`);
 *       const { read, write } = this.pp.get(ctx, w, h, f);
 *       // f.frameIndex == 0: no valid history (read is all zero); otherwise read holds the previous frame's write.
 *   }
 */

import {
    ResourceBindFlags,
    ResourceFormat,
    type Device,
    type RenderContext,
    type RenderGraph,
    type Scene,
    type Texture,
} from "@web-falcor/falcor";

export interface HistoryFrame {
    /** Frames since the last reset: 0 on the first frame after a reset (no valid history). */
    frameIndex: number;
    /** True when the history was reset at the start of this frame. */
    reset: boolean;
    /** Why it was reset ("" when it was not): fresh | explicit | scene | camera | key | camera-reset | camera-cut. */
    reason: string;
}

function sameMatrix(a: ArrayLike<number>, b: ArrayLike<number>): boolean {
    for (let i = 0; i < 16; i++) if (a[i] !== b[i]) return false;
    return true;
}

export class HistoryTracker {
    private frame = 0;
    private pending = "fresh";
    private scene: Scene | null = null;
    private camera: unknown = null;
    private epoch = 0;
    private key = "";
    private lastViewProj: Float32Array | null = null;

    /** Requests a reset at the next begin(). */
    reset(reason = "explicit"): void {
        this.pending ||= reason;
    }

    /** Frames since the last reset as of the last begin() (0 before the first one). */
    get frameIndex(): number {
        return this.frame;
    }

    /** Call once at the top of execute(). */
    begin(scene: Scene | null, key = ""): HistoryFrame {
        const cam = scene?.camera ?? null;
        let reason = this.pending;
        let cur: Float32Array | null = null;
        if (cam) {
            const d = cam.getData();
            cur = new Float32Array(d.viewProjMatNoJitter.toArray());
            if (!reason) {
                if (scene !== this.scene) reason = "scene";
                else if (cam !== this.camera) reason = "camera";
                else if (key !== this.key) reason = "key";
                else if (cam.historyEpoch !== this.epoch) reason = "camera-reset";
                else if (this.lastViewProj && !sameMatrix(this.lastViewProj, cur) && sameMatrix(d.prevViewProjMatNoJitter.toArray(), cur)) reason = "camera-cut";
            }
        } else if (!reason && (scene !== this.scene || key !== this.key)) {
            reason = scene !== this.scene ? "scene" : "key";
        }
        this.scene = scene;
        this.camera = cam;
        this.epoch = cam?.historyEpoch ?? 0;
        this.key = key;
        this.lastViewProj = cur;
        this.pending = "";
        this.frame = reason ? 0 : this.frame + 1;
        return { frameIndex: this.frame, reset: !!reason, reason };
    }
}

export class PingPongTextures {
    private tex: [Texture, Texture] | null = null;

    constructor(
        private readonly device: Device,
        readonly format: ResourceFormat = ResourceFormat.RGBA32Float,
        readonly name = "history",
    ) {}

    /** The pair for this frame; reallocates on a size change and clears both on a reset or reallocation. */
    get(ctx: RenderContext, width: number, height: number, frame: HistoryFrame): { read: Texture; write: Texture } {
        let fresh = false;
        if (!this.tex || this.tex[0].width !== width || this.tex[0].height !== height) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget;
            this.tex = [0, 1].map(() => this.device.createTexture2D(width, height, this.format, 1, 1, undefined, flags)) as [Texture, Texture];
            fresh = true;
        }
        if (fresh || frame.reset) for (const t of this.tex) ctx.clearTexture(t, [0, 0, 0, 0]);
        const w = frame.frameIndex & 1;
        return { write: this.tex[w]!, read: this.tex[1 - w]! };
    }

    /** Both textures (for tests and debugging); null before the first get(). */
    get textures(): readonly Texture[] | null {
        return this.tex;
    }
}

/** Calls resetHistory() on every pass of `graph` that has one. */
export function resetGraphHistory(graph: RenderGraph): void {
    for (const { pass } of graph.getPasses()) (pass as { resetHistory?: () => void }).resetHistory?.();
}
