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
 *   private mHistory: HistoryTracker = new HistoryTracker();
 *   private mHistoryTextures: PingPongTextures = new PingPongTextures(device, ResourceFormat.RGBA32Float, "MyPass::history");
 *   resetHistory() { this.mHistory.reset(); }
 *   execute(ctx, rd) {
 *       const f = this.mHistory.begin(this.scene, `${w}x${h}`);
 *       const { read, write } = this.mHistoryTextures.get(ctx, w, h, f);
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

/** History validity, reset reason and frame index returned by HistoryTracker.begin(). */
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

/** Tracks scene, camera and layout changes so temporal passes discard stale state. */
export class HistoryTracker {
    private mFrame: number = 0;
    private mPending: string = "fresh";
    private mScene: Scene | null = null;
    private mCamera: unknown = null;
    private mEpoch: number = 0;
    private mKey: string = "";
    private mLastViewProj: Float32Array | null = null;

    /** Requests a reset at the next begin(). */
    reset(reason = "explicit"): void {
        this.mPending ||= reason;
    }

    /** Frames since the last reset as of the last begin() (0 before the first one). */
    get frameIndex(): number {
        return this.mFrame;
    }

    /** Call once at the top of execute(). */
    begin(scene: Scene | null, key = ""): HistoryFrame {
        const cam = scene?.camera ?? null;
        let reason = this.mPending;
        let cur: Float32Array | null = null;
        if (cam) {
            const d = cam.getData();
            cur = new Float32Array(d.viewProjMatNoJitter.toArray());
            if (!reason) {
                if (scene !== this.mScene) reason = "scene";
                else if (cam !== this.mCamera) reason = "camera";
                else if (key !== this.mKey) reason = "key";
                else if (cam.historyEpoch !== this.mEpoch) reason = "camera-reset";
                else if (this.mLastViewProj && !sameMatrix(this.mLastViewProj, cur) && sameMatrix(d.prevViewProjMatNoJitter.toArray(), cur))
                    reason = "camera-cut";
            }
        } else if (!reason && (scene !== this.mScene || key !== this.mKey)) {
            reason = scene !== this.mScene ? "scene" : "key";
        }
        this.mScene = scene;
        this.mCamera = cam;
        this.mEpoch = cam?.historyEpoch ?? 0;
        this.mKey = key;
        this.mLastViewProj = cur;
        this.mPending = "";
        this.mFrame = reason ? 0 : this.mFrame + 1;
        return { frameIndex: this.mFrame, reset: !!reason, reason };
    }
}

/** Owns a texture pair that alternates read/write roles and clears both textures after a history reset. */
export class PingPongTextures {
    private mTextures: [Texture, Texture] | null = null;

    constructor(
        private readonly device: Device,
        readonly format: ResourceFormat = ResourceFormat.RGBA32Float,
        readonly name = "history",
    ) {}

    /** The pair for this frame; reallocates on a size change and clears both on a reset or reallocation. */
    get(ctx: RenderContext, width: number, height: number, frame: HistoryFrame): { read: Texture; write: Texture } {
        let fresh = false;
        if (!this.mTextures || this.mTextures[0].width !== width || this.mTextures[0].height !== height) {
            const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget;
            this.mTextures = [0, 1].map(() => this.device.createTexture2D(width, height, this.format, 1, 1, undefined, flags)) as [
                Texture,
                Texture,
            ];
            fresh = true;
        }
        if (fresh || frame.reset) for (const t of this.mTextures) ctx.clearTexture(t, [0, 0, 0, 0]);
        const w = frame.frameIndex & 1;
        return { write: this.mTextures[w]!, read: this.mTextures[1 - w]! };
    }

    /** Both textures (for tests and debugging); null before the first get(). */
    get textures(): readonly Texture[] | null {
        return this.mTextures;
    }
}

/** Calls resetHistory() on every pass of `graph` that has one. */
export function resetGraphHistory(graph: RenderGraph): void {
    for (const { pass } of graph.getPasses()) (pass as { resetHistory?: () => void }).resetHistory?.();
}
