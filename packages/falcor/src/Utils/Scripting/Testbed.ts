/**
 * Mirrors Falcor/Core/Testbed: the Python-driven application (`falcor.Testbed`) behind
 * Falcor's Python examples (scripts/python). Each frame ticks the clock, clears the
 * target, runs the render graph and blits its first output, blits the render texture,
 * and presents. `runTestbedScript` runs an unmodified script against it.
 *
 * §9: the window is an optional canvas; Python's blocking `while not
 * testbed.should_close: testbed.frame()` loops yield to the browser inside frame()
 * (JSPI through pyodide's run_sync), and GPU readbacks (to_numpy) wait the same way.
 * Headless runs close after `maxFrames` frames. Escape closes the window.
 */

import type { Device } from "../../Core/API/Device.js";
import { Fbo, FboAttachmentType } from "../../Core/API/FBO.js";
import { ResourceFormat } from "../../Core/API/Formats.js";
import { presentToCanvas } from "../../Core/API/Present.js";
import type { Texture } from "../../Core/API/Texture.js";
import { RenderGraph } from "../../RenderGraph/RenderGraph.js";
import type { Scene } from "../../Scene/Scene.js";
import { AssetCategory, AssetResolver } from "../../Core/AssetResolver.js";
import { Clock } from "../Timing/Clock.js";
import { FrameRate } from "../Timing/FrameRate.js";
import { PyUiScreen } from "../UI/PythonUI.js";
import { TextRenderer } from "../UI/TextRenderer.js";
import { runSceneScript } from "./Scripting.js";

export interface TestbedOptions {
    width?: number;
    height?: number;
    createWindow?: boolean;
    title?: string;
    showFPS?: boolean;
    /** Web: the canvas standing in for the window (createWindow). */
    canvas?: HTMLCanvasElement | null;
    /** Web: close after this many frames (headless runs). */
    maxFrames?: number;
    /** Web: the element `falcor.ui` windows overlay (default: the canvas's parent). */
    uiHost?: HTMLElement | null;
    /** Web: called after each frame (tests drive the UI from here). */
    onFrame?: (testbed: Testbed, frame: number) => void;
}

export class Testbed {
    readonly clock = new Clock();
    readonly frameRate = new FrameRate();
    renderGraph: RenderGraph | null = null;
    renderTexture: Texture | null = null;
    scene: Scene | null = null;
    /** Directory of the loaded scene file (relative texture paths resolve against it). */
    sceneBaseUrl = "";
    showUI = true;
    private targetFbo: Fbo;
    private context: GPUCanvasContext | null = null;
    private canvasFormat: GPUTextureFormat = "bgra8unorm";
    private closeRequested = false;
    private graphNeedsInit = false;
    private textRenderer: TextRenderer | null = null;
    private frames = 0;
    private uiScreen: PyUiScreen | null = null;

    constructor(
        readonly device: Device,
        private readonly options: TestbedOptions = {},
    ) {
        const width = options.width ?? 1920;
        const height = options.height ?? 1080;
        this.targetFbo = Fbo.create2D(device, width, height, ResourceFormat.RGBA8UnormSrgb, ResourceFormat.D32Float);
        const canvas = options.createWindow ? options.canvas : null;
        if (canvas) {
            canvas.width = width;
            canvas.height = height;
            if (options.title) document.title = options.title;
            this.context = canvas.getContext("webgpu");
            this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context?.configure({ device: device.gpuDevice, format: this.canvasFormat });
            window.addEventListener("keydown", (e) => {
                if (e.key === "Escape") this.closeRequested = true;
            });
        }
        if (options.showFPS ?? true) {
            this.textRenderer = new TextRenderer(device);
            void this.textRenderer.init();
        }
    }

    /** Mirrors Testbed::shouldClose. */
    get shouldClose(): boolean {
        return this.closeRequested || (this.options.maxFrames !== undefined && this.frames >= this.options.maxFrames);
    }

    /** Mirrors Testbed::getScreen: the root of `falcor.ui` widgets. */
    get screen(): PyUiScreen {
        if (!this.uiScreen) {
            const host = this.options.uiHost ?? (this.options.createWindow ? this.options.canvas?.parentElement : null) ?? document.createElement("div");
            this.uiScreen = new PyUiScreen(host);
        }
        return this.uiScreen;
    }

    /** The target FBO (native: the window's frame buffer). */
    getTargetFbo(): Fbo {
        return this.targetFbo;
    }

    getFrameCount(): number {
        return this.frames;
    }

    setRenderGraph(graph: RenderGraph | null): void {
        this.renderGraph = graph;
        if (graph) {
            graph.onResize(this.targetFbo.width, this.targetFbo.height);
            if (this.scene) graph.setScene(this.scene);
            this.graphNeedsInit = true;
        }
    }

    /** Mirrors Testbed::createRenderGraph. */
    createRenderGraph(name = ""): RenderGraph {
        return new RenderGraph(this.device, name);
    }

    /** Mirrors Testbed::resizeFrameBuffer. */
    resizeFrameBuffer(width: number, height: number): void {
        this.targetFbo = Fbo.create2D(this.device, width, height, ResourceFormat.RGBA8UnormSrgb, ResourceFormat.D32Float);
        const canvas = this.options.canvas;
        if (this.context && canvas) {
            canvas.width = width;
            canvas.height = height;
        }
        this.renderGraph?.onResize(width, height);
        this.graphNeedsInit = true;
        this.scene?.camera.setAspectRatio(width / height);
    }

    /** Mirrors Testbed::loadScene (paths resolve through the asset resolver, like native). */
    async loadScene(path: string, buildFlags = 0): Promise<void> {
        const url = path.startsWith("/") ? path : await AssetResolver.getDefaultResolver().resolvePath(path, AssetCategory.Scene);
        const baseUrl = url.slice(0, url.lastIndexOf("/"));
        this.sceneBaseUrl = baseUrl;
        this.scene = await runSceneScript(this.device, await (await fetch(url)).text(), baseUrl, { flags: buildFlags });
        if (this.renderGraph) {
            this.renderGraph.setScene(this.scene);
            this.graphNeedsInit = true;
        }
    }

    /** Mirrors Testbed::frame; resolves once the browser has shown the frame (next animation frame with a canvas). */
    async frame(): Promise<void> {
        const ctx = this.device.renderContext;
        this.clock.tick();
        this.frameRate.newFrame();
        ctx.clearFbo(this.targetFbo, [1, 0, 1, 1], 1, 0, FboAttachmentType.All);
        if (this.renderGraph) {
            if (this.graphNeedsInit) {
                await this.renderGraph.init();
                this.graphNeedsInit = false;
            }
            if (this.scene?.isAnimated()) this.scene.animate(this.clock.getTime());
            this.renderGraph.execute(ctx);
            const names = this.renderGraph.getOutputNames();
            const out = names.length > 0 ? this.renderGraph.getOutput(names[0]!) : undefined;
            if (out) ctx.blit(out, this.targetFbo.getColorTexture(0)!);
        }
        if (this.renderTexture) ctx.blit(this.renderTexture, this.targetFbo.getColorTexture(0)!);
        if (this.showUI && this.textRenderer?.isReady()) this.textRenderer.render(ctx, this.frameRate.getMsg(), this.targetFbo, [10, 10]);
        this.device.profilerHook?.endFrame(ctx.getEncoder());
        if (this.context) presentToCanvas(this.device, this.targetFbo.getColorTexture(0)!, this.context.getCurrentTexture(), this.canvasFormat);
        ctx.submit();
        this.frames++;
        if (this.uiScreen) this.uiScreen.root.style.display = this.showUI ? "" : "none";
        this.options.onFrame?.(this, this.frames);
        if (this.context) await new Promise((resolve) => requestAnimationFrame(resolve));
        else await this.device.gpuDevice.queue.onSubmittedWorkDone();
    }

    /** Mirrors Testbed::run: frames until the window closes. */
    async run(): Promise<void> {
        while (!this.shouldClose) await this.frame();
    }

    /** Mirrors Testbed::captureOutput: the graph output as a file (downloaded in a page). */
    async captureOutput(path: string, outputIndex = 0): Promise<Uint8Array | null> {
        const names = this.renderGraph?.getOutputNames() ?? [];
        const tex = names[outputIndex] ? this.renderGraph!.getOutput(names[outputIndex]!) : undefined;
        if (!tex) return null;
        const { Bitmap } = await import("../Image/Bitmap.js");
        const ext = path.slice(path.lastIndexOf(".") + 1);
        return tex.captureToFile(0, 0, path, Bitmap.getFormatFromFileExtension(ext), undefined, typeof document !== "undefined");
    }
}
