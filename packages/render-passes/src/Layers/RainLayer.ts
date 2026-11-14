/**
 * RainLayer: falling rain streaks rendered as a colour + depth layer for DepthComposite.
 *
 * `count` drops live in a cube of edge `box` centred on the camera. Drop i starts at a seed-derived point of the cube
 * and falls along -upAxis at `speed` (x (1 +- speedJitter/2)) plus `wind`; its position at `time` is wrapped into the
 * cube (modulo the edge), so the rain is world-anchored (parallax under camera motion) and a pure function of `time`,
 * the camera position and `seed`: the pass keeps no state. Set `time` per frame for clips.
 *
 * Each drop is drawn as a streak from its position back along its velocity over `exposure` seconds (motion blur of a
 * camera shutter), at most `maxLengthPx` pixels long, `width` pixels wide (anti-aliased), brighter at the head, faded
 * towards the cube faces (no popping when a drop wraps) and hidden per pixel behind the scene (`posW`). Contributions are
 * accumulated with integer atomics, so the result does not depend on thread order.
 *   color: rgb = colour x intensity x sum of weights, a = min(1, sum of weights), weight = opacity x coverage x taper;
 *          composite with DepthComposite (premultiplied; additive for a glowing look);
 *   depth: r = view depth of the nearest streak (DepthComposite's posW convention), 3.4e38 where none.
 *
 * Sky occlusion (`skyOcclusion`): "None" (default) draws every drop. "RayTraced" casts one visibility ray per drop from
 * its head along +upAxis (up to `skyOcclusionDistance`) through the scene's software BVH (shared visibility service,
 * Visibility.slangh); drops with geometry above them (under a roof, an awning or a tree) are not drawn, so interiors stay
 * dry. Drops with open sky above them are drawn exactly as with "None".
 *
 * Identity: count 0, opacity 0 (or intensity 0 and opacity 0) give an empty layer, which DepthComposite passes through
 * bit for bit.
 *
 * Inputs : posW? (occlusion)
 * Outputs: color, depth (RGBA32Float)
 */

import {
    Buffer,
    ComputePass,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";
import { addColorOutput, addTexInput, dummyTexture, parseColour } from "../Stylize/StyleCommon.js";

const kShaderFile = "RenderPasses/Layers/RainLayer.cs.slang";
const kEntries = ["clearAccum", "splat", "resolve"] as const;
type Entry = (typeof kEntries)[number];

const kDefaults = {
    count: 30000,
    seed: 0,
    box: 16,
    speed: 8,
    speedJitter: 0.3,
    wind: [0, 0, 0] as number[],
    exposure: 0.02,
    width: 1,
    maxLengthPx: 48,
    colour: [0.75, 0.8, 0.9] as number[] | string,
    intensity: 1,
    opacity: 0.35,
    near: 0.2,
    time: 0,
    upAxis: [0, 1, 0] as number[],
    skyOcclusion: "None" as RainSkyOcclusion,
    skyOcclusionDistance: 1e30,
};
export const kRainSkyOcclusionModes = ["None", "RayTraced"] as const;
export type RainSkyOcclusion = (typeof kRainSkyOcclusionModes)[number];
export type RainLayerParams = typeof kDefaults;

export class RainLayer extends RenderPass {
    private p: RainLayerParams = JSON.parse(JSON.stringify(kDefaults));
    private passes = new Map<Entry, ComputePass>();
    private splatKey = "";
    private accum: Buffer | null = null;
    private depthBits: Buffer | null = null;

    constructor(device: Device, props: Properties) {
        super(device);
        for (const k of Object.keys(kDefaults) as (keyof RainLayerParams)[]) (this.p as Record<string, unknown>)[k] = props.get(k, kDefaults[k] as never);
        this.p.count = Math.max(0, Math.min(Math.round(this.p.count), 1 << 20));
        if (!(kRainSkyOcclusionModes as readonly string[]).includes(this.p.skyOcclusion)) {
            throw new Error(`RainLayer: unknown skyOcclusion '${this.p.skyOcclusion}' (${kRainSkyOcclusionModes.join(", ")})`);
        }
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.passes.delete("splat");
        this.splatKey = "";
    }

    override getProperties(): Properties {
        return new Properties(JSON.parse(JSON.stringify(this.p)));
    }

    override renderUI(ui: UIWidgets): void {
        ui.slider("Opacity", this.p.opacity, 0, 1, 0.01, (v) => (this.p.opacity = v));
        ui.slider("Intensity", this.p.intensity, 0, 10, 0.01, (v) => (this.p.intensity = v));
        ui.slider("Speed", this.p.speed, 0, 20, 0.1, (v) => (this.p.speed = v));
        ui.slider("Time", this.p.time, 0, 60, 0.01, (v) => (this.p.time = v));
        ui.dropdown("Sky occlusion", [...kRainSkyOcclusionModes], this.p.skyOcclusion, (v) => (this.p.skyOcclusion = v as RainSkyOcclusion));
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        addTexInput(r, "posW", "Scene world position (occlusion)", true);
        addColorOutput(r, "color", "Rain layer: streak radiance (rgb), coverage (a)", w, h);
        addColorOutput(r, "depth", "Nearest streak view depth (r)", w, h, true);
        return r;
    }

    /** Sky occlusion needs the scene (BVH); without one the pass draws every drop. */
    private occluded(): boolean {
        return this.p.skyOcclusion === "RayTraced" && !!this.scene;
    }

    private kernel(e: Entry): ComputePass {
        // Only the splat kernel traces rays: with sky occlusion it gets the scene defines and the visibility service in
        // BVH mode; otherwise (and for the other kernels) the plain shader without defines.
        const defines = e === "splat" && this.occluded() ? this.scene!.getSceneDefines().addAll({ RAIN_SKY_OCCLUSION: 1, VIS_MODE: 2 }) : null;
        const key = defines ? defines.key() : "";
        let k = this.passes.get(e);
        if (!k || (e === "splat" && key !== this.splatKey)) {
            k = defines ? ComputePass.create(this.device, { path: kShaderFile, csEntry: e, defines }) : ComputePass.create(this.device, { path: kShaderFile, csEntry: e });
            this.passes.set(e, k);
            if (e === "splat") this.splatKey = key;
        }
        return k;
    }

    override execute(ctx: RenderContext, rd: RenderData): void {
        const color = rd.getTexture("color");
        if (!color) return;
        const W = color.width, H = color.height;
        const flags = ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess;
        if (!this.accum || this.accum.size !== W * H * 16) {
            this.accum = new Buffer(this.device, { size: W * H * 16, structSize: 4, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "RainLayer::accum" });
            this.depthBits = new Buffer(this.device, { size: W * H * 4, structSize: 4, bindFlags: flags, memoryType: MemoryType.DeviceLocal, name: "RainLayer::depth" });
        }
        const posW = rd.getTexture("posW"), depthOut = rd.getTexture("depth");
        const cam = this.scene?.camera;
        const vp = cam?.getData().viewProjMatNoJitter;
        const rows = [0, 1, 2, 3].map((r) => (vp ? [vp.get(r, 0), vp.get(r, 1), vp.get(r, 2), vp.get(r, 3)] : [0, 0, 0, 0]));
        const pos = cam ? cam.getPosition() : { x: 0, y: 0, z: 0 };
        const tgt = cam ? cam.getTarget() : { x: 0, y: 0, z: -1 };
        const f = [tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z];
        const fl = Math.hypot(f[0]!, f[1]!, f[2]!) || 1;
        const u = this.p.upAxis.map(Number);
        const ul = Math.hypot(u[0]!, u[1]!, u[2]!) || 1;
        const count = cam ? this.p.count : 0;
        const consts: Record<string, unknown> = {
            gFrameDim: [W, H], gCount: count, gSeed: Math.max(0, Math.round(this.p.seed)) >>> 0,
            gUp: u.map((x) => x / ul), gBox: Math.max(Number(this.p.box), 1e-3), gWind: this.p.wind.map(Number), gSpeed: Number(this.p.speed),
            gColour: parseColour(this.p.colour), gIntensity: Number(this.p.intensity), gCamPos: [pos.x, pos.y, pos.z], gTime: Number(this.p.time),
            gCamFwd: [f[0]! / fl, f[1]! / fl, f[2]! / fl], gExposure: Math.max(Number(this.p.exposure), 0), gHalfWidth: Math.max(Number(this.p.width), 0.1) * 0.5,
            gOpacity: Math.max(Number(this.p.opacity), 0), gMaxLenPx: Math.max(Number(this.p.maxLengthPx), 1), gNear: Math.max(Number(this.p.near), 1e-3),
            gHasPosW: posW ? 1 : 0, gSpeedJitter: Math.min(Math.max(Number(this.p.speedJitter), 0), 1.9),
            gViewProj0: rows[0], gViewProj1: rows[1], gViewProj2: rows[2], gViewProj3: rows[3],
        };
        const run = (e: Entry, x: number, y: number) => {
            const k = this.kernel(e);
            const root = k.getRootVar();
            const cb = root["CB"]!;
            for (const [name, v] of Object.entries(consts)) cb[name] = v as never;
            root["gAccum"] = this.accum!;
            root["gDepthBits"] = this.depthBits!;
            root["gPosW"] = posW ?? dummyTexture(this.device);
            root["gColor"] = color;
            root["gDepth"] = depthOut ?? dummyTexture(this.device, "out:depth");
            if (e === "splat" && this.occluded()) {
                this.scene!.bindShaderData(root);
                root["SkyCB"]!["gSkyDist"] = Math.max(Number(this.p.skyOcclusionDistance), 1e-3);
            }
            k.execute(ctx, x, y);
        };
        run("clearAccum", W, H);
        if (count > 0 && Number(this.p.opacity) > 0) run("splat", count, 1);
        run("resolve", W, H);
    }
}

registerRenderPass("RainLayer", (device, props) => new RainLayer(device, props), {
    desc: "Falling rain streaks (camera-centred, world-anchored, time-driven) as a colour + depth layer for DepthComposite",
});
