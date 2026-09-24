/**
 * Mirrors Falcor/Core/SampleApp: the base class of Falcor's sample applications. Owns the
 * device, a target FBO (color + depth), the global clock, frame rate and text renderer,
 * and drives the onLoad/onResize/onFrameRender/onGuiRender/on*Event callbacks.
 *
 * §9: the window is a canvas (the FBO is presented to it every frame through
 * requestAnimationFrame), the Gui a DOM panel of UIWidgets rebuilt when a control
 * changes, screen captures download, and shader reload is F6 (the browser owns F5).
 * Headless apps (config.headless, no canvas) render only when renderFrame() is called,
 * which is how tests drive them. PixelZoom and gamepad input are not ported.
 */

import { Device, type DeviceDesc } from "./API/Device.js";
import { Fbo } from "./API/FBO.js";
import { ResourceFormat } from "./API/Formats.js";
import { presentToCanvas } from "./API/Present.js";
import type { RenderContext } from "./API/RenderContext.js";
import type { Texture } from "./API/Texture.js";
import { BitmapFileFormat } from "../Utils/Image/Bitmap.js";
import { Clock } from "../Utils/Timing/Clock.js";
import { FrameRate } from "../Utils/Timing/FrameRate.js";
import { TextRenderer } from "../Utils/UI/TextRenderer.js";
import { DomWidgets } from "../Utils/UI/DomWidgets.js";
import type { UIWidgets } from "../RenderGraph/UIWidgets.js";
import {
    KeyboardEventType,
    ModifierFlags,
    MouseEventType,
    toKeyboardEvent,
    toMouseEvent,
    type KeyboardEvent,
    type MouseEvent,
} from "../Utils/UI/InputTypes.js";
import { fetchShaderSources, initProgramSystem } from "./Program/ShaderSources.js";
import { getGlobalSettings } from "../Utils/Scripting/Scripting.js";
import type { Settings } from "../Utils/Settings.js";
import { Logger } from "../Utils/Logger.js";

/** Mirrors HotReloadFlags. */
export enum HotReloadFlags {
    None = 0,
    Program = 1,
}

/** Mirrors Window::Desc (the subset a canvas has). */
export interface WindowDesc {
    title?: string;
    width?: number;
    height?: number;
    resizableWindow?: boolean;
    enableVSync?: boolean;
}

/** Mirrors SampleAppConfig. */
export interface SampleAppConfig {
    deviceDesc?: DeviceDesc;
    /** Use an existing device (e.g. the test harness's) instead of creating one. */
    device?: Device;
    windowDesc?: WindowDesc;
    colorFormat?: ResourceFormat;
    depthFormat?: ResourceFormat;
    headless?: boolean;
    timeScale?: number;
    pauseTime?: boolean;
    showUI?: boolean;
    /** Web: the canvas standing in for the window, and the element hosting the Gui panel. */
    canvas?: HTMLCanvasElement;
    uiContainer?: HTMLElement;
}

const kKeyboardShortcuts =
    "ESC - Quit\n" +
    "F2 - Show/hide UI\n" +
    "F6 - Reload shaders\n" +
    "F12 - Capture screenshot\n" +
    "V - Toggle VSync\n" +
    "Pause|Space - Pause/resume the global timer\n" +
    "Ctrl+Pause|Space - Pause/resume the renderer\n";

export abstract class SampleApp {
    private device: Device | null = null;
    private targetFbo: Fbo | null = null;
    private pausedRenderOutput: Texture | null = null;
    private textRenderer: TextRenderer | null = null;
    private readonly clock = new Clock();
    private readonly frameRate = new FrameRate();
    private readonly config: Required<Pick<SampleAppConfig, "colorFormat" | "depthFormat" | "headless" | "timeScale" | "pauseTime" | "showUI">> & SampleAppConfig;
    private context: GPUCanvasContext | null = null;
    private canvasFormat: GPUTextureFormat = "bgra8unorm";
    private shouldTerminate = false;
    private returnCode = 0;
    private rendererPaused = false;
    private vsyncOn: boolean;
    private showUI: boolean;
    private captureScreenRequested = false;
    private uiDirty = true;
    private lastUiBuild = 0;
    private removeListeners: (() => void)[] = [];

    constructor(config: SampleAppConfig = {}) {
        this.config = {
            colorFormat: ResourceFormat.BGRA8UnormSrgb,
            depthFormat: ResourceFormat.D32Float,
            headless: false,
            timeScale: 1,
            pauseTime: false,
            showUI: true,
            ...config,
        };
        this.vsyncOn = config.windowDesc?.enableVSync ?? false;
        this.showUI = this.config.showUI;
    }

    // Callbacks (mirroring SampleApp's virtuals).
    onLoad(_renderContext: RenderContext): void | Promise<void> {}
    onShutdown(): void {}
    onResize(_width: number, _height: number): void {}
    onFrameRender(_renderContext: RenderContext, _targetFbo: Fbo): void {}
    onGuiRender(_gui: UIWidgets): void {}
    onOptionsChange(): void {}
    onHotReload(_reloaded: HotReloadFlags): void {}
    onKeyEvent(_keyEvent: KeyboardEvent): boolean {
        return false;
    }
    onMouseEvent(_mouseEvent: MouseEvent): boolean {
        return false;
    }

    getSettings(): Settings {
        return getGlobalSettings();
    }
    getDevice(): Device {
        if (!this.device) throw new Error("SampleApp: not initialized");
        return this.device;
    }
    getRenderContext(): RenderContext {
        return this.getDevice().renderContext;
    }
    getTargetFbo(): Fbo {
        if (!this.targetFbo) throw new Error("SampleApp: not initialized");
        return this.targetFbo;
    }
    getTextRenderer(): TextRenderer {
        return this.textRenderer!;
    }
    getGlobalClock(): Clock {
        return this.clock;
    }
    getFrameRate(): FrameRate {
        return this.frameRate;
    }
    getConfig(): SampleAppConfig {
        return { ...this.config, showUI: this.showUI };
    }
    toggleUI(showUI: boolean): void {
        this.showUI = showUI;
        this.uiDirty = true;
    }
    isUiEnabled(): boolean {
        return this.showUI;
    }
    pauseRenderer(pause: boolean): void {
        this.rendererPaused = pause;
    }
    isRendererPaused(): boolean {
        return this.rendererPaused;
    }
    toggleVsync(on: boolean): void {
        this.vsyncOn = on;
    }
    isVsyncEnabled(): boolean {
        return this.vsyncOn;
    }
    static getKeyboardShortcutsStr(): string {
        return kKeyboardShortcuts;
    }

    /** Creates the device, loads the shader tree and fonts, calls onLoad, sizes the frame buffer. */
    async initialize(): Promise<void> {
        const device = this.config.device ?? (await Device.create(this.config.deviceDesc));
        this.device = device;
        if (!this.config.device) await initProgramSystem(device);
        this.textRenderer = new TextRenderer(device);
        await this.textRenderer.init();
        this.clock.setTimeScale(this.config.timeScale);
        if (this.config.pauseTime) this.clock.pause();

        const canvas = this.config.canvas;
        if (!this.config.headless && canvas) {
            if (this.config.windowDesc?.title) document.title = this.config.windowDesc.title;
            this.context = canvas.getContext("webgpu");
            if (!this.context) throw new Error("SampleApp: no webgpu canvas context");
            this.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
            this.context.configure({ device: device.gpuDevice, format: this.canvasFormat });
            canvas.width = this.config.windowDesc?.width ?? canvas.clientWidth ?? 1920;
            canvas.height = this.config.windowDesc?.height ?? canvas.clientHeight ?? 1080;
            this.attachInput(canvas);
        }
        const width = canvas?.width ?? this.config.windowDesc?.width ?? 1920;
        const height = canvas?.height ?? this.config.windowDesc?.height ?? 1080;
        this.resizeTargetFbo(width, height);
        await this.onLoad(device.renderContext);
        this.onResize(width, height);
    }

    /** Mirrors SampleApp::run: initializes, then renders frames until shutdown(). */
    async run(): Promise<number> {
        await this.initialize();
        if (this.config.headless || !this.context) {
            while (!this.shouldTerminate) {
                this.renderFrame();
                await this.device!.gpuDevice.queue.onSubmittedWorkDone();
            }
        } else {
            await new Promise<number>((resolve) => {
                const loop = () => {
                    if (this.shouldTerminate) return resolve(this.returnCode);
                    this.renderFrame();
                    requestAnimationFrame(loop);
                };
                requestAnimationFrame(loop);
            });
        }
        this.onShutdown();
        for (const remove of this.removeListeners) remove();
        return this.returnCode;
    }

    /** Mirrors SampleApp::shutdown. */
    shutdown(returnCode = 0): void {
        this.shouldTerminate = true;
        this.returnCode = returnCode;
    }

    /** Mirrors SampleApp::resizeFrameBuffer (the canvas follows). */
    resizeFrameBuffer(width: number, height: number): void {
        if (this.config.canvas && this.context) {
            this.config.canvas.width = width;
            this.config.canvas.height = height;
        }
        this.resizeTargetFbo(width, height);
        this.onResize(width, height);
    }

    private resizeTargetFbo(width: number, height: number): void {
        this.targetFbo = Fbo.create2D(this.getDevice(), width, height, this.config.colorFormat, this.config.depthFormat);
    }

    /** Mirrors SampleApp::renderFrame. */
    renderFrame(): void {
        const device = this.getDevice();
        const ctx = device.renderContext;
        // Check clock exit condition.
        if (this.clock.shouldExit()) this.shutdown();
        this.clock.tick();
        this.frameRate.newFrame();
        const target = this.getTargetFbo();
        if (this.rendererPaused && this.pausedRenderOutput) {
            ctx.blit(this.pausedRenderOutput, target.getColorTexture(0)!);
        } else {
            this.onFrameRender(ctx, target);
            if (this.rendererPaused) {
                const src = target.getColorTexture(0)!;
                this.pausedRenderOutput = device.createTexture2D(src.width, src.height, src.format, 1, 1);
                ctx.copyTexture(this.pausedRenderOutput, src);
            } else {
                this.pausedRenderOutput = null;
            }
        }
        this.renderUI();
        device.profilerHook?.endFrame(ctx.getEncoder());
        if (this.captureScreenRequested) this.captureScreen(target.getColorTexture(0)!);
        if (this.context) presentToCanvas(device, target.getColorTexture(0)!, this.context.getCurrentTexture(), this.canvasFormat);
        ctx.submit();
    }

    /** Mirrors SampleApp::renderGlobalUI: shortcuts, clock and renderer controls. */
    renderGlobalUI(ui: UIWidgets): void {
        ui.text(this.frameRate.getMsg(this.vsyncOn));
        const g = ui.group("Global Controls");
        g.slider("Time", this.clock.getTime(), 0, Math.max(60, this.clock.getTime() * 2), 0.01, (t) => this.clock.setTime(t));
        g.button("Reset", () => this.clock.setTime(0));
        g.button(this.clock.isPaused() ? "Play" : "Pause", () => (this.clock.isPaused() ? this.clock.play() : this.clock.pause()));
        g.button("Stop", () => this.clock.stop());
        g.slider("Scale", this.clock.getTimeScale(), 0, 10, 0.01, (s) => this.clock.setTimeScale(s));
        g.button(this.rendererPaused ? "Resume Rendering" : "Pause Rendering", () => (this.rendererPaused = !this.rendererPaused));
        g.button("Screen Capture", () => (this.captureScreenRequested = true));
        const help = ui.group("Keyboard Shortcuts");
        for (const line of kKeyboardShortcuts.trim().split("\n")) help.text(line);
    }

    private renderUI(): void {
        const container = this.config.uiContainer;
        if (!container) return;
        // The DOM panel is retained: rebuild it on changes, and twice a second for live
        // values (frame rate, time) unless the user is working a control.
        const now = performance.now();
        const interacting = container.matches(":hover") && (document.activeElement === null || container.contains(document.activeElement));
        if (!this.uiDirty && (now - this.lastUiBuild < 500 || interacting)) return;
        this.uiDirty = false;
        this.lastUiBuild = now;
        container.replaceChildren();
        container.hidden = !this.showUI;
        if (this.showUI) this.onGuiRender(new DomWidgets(container, () => (this.uiDirty = true)));
    }

    private captureScreen(texture: Texture): void {
        this.captureScreenRequested = false;
        const name = `${(this.config.windowDesc?.title ?? "SampleApp").replace(/\W+/g, "")}.png`;
        void texture.captureToFile(0, 0, name, BitmapFileFormat.PngFile);
    }

    private async reloadShaders(): Promise<void> {
        const sources = await fetchShaderSources();
        this.getDevice().programManager.reloadAllPrograms({ resolveSource: (p) => sources.get(p), filePaths: [...sources.keys()] });
        this.onHotReload(HotReloadFlags.Program);
        Logger.info("Shaders reloaded.");
    }

    /** Mirrors SampleApp::handleKeyboardEvent (after the app's own onKeyEvent). */
    handleKeyboardEvent(keyEvent: KeyboardEvent): void {
        if (this.onKeyEvent(keyEvent)) return;
        if (keyEvent.type !== KeyboardEventType.KeyPressed) return;
        if (keyEvent.mods & ModifierFlags.Ctrl) {
            if (keyEvent.key === "Pause" || keyEvent.key === "Space") this.rendererPaused = !this.rendererPaused;
        } else if (keyEvent.mods === ModifierFlags.None) {
            switch (keyEvent.key) {
                case "F12":
                    this.captureScreenRequested = true;
                    break;
                case "V":
                    this.vsyncOn = !this.vsyncOn;
                    this.frameRate.reset();
                    this.clock.setTime(0);
                    break;
                case "F2":
                    this.toggleUI(!this.showUI);
                    break;
                case "F6":
                    void this.reloadShaders();
                    break;
                case "Escape":
                    this.shutdown();
                    break;
                case "Pause":
                case "Space":
                    this.clock.isPaused() ? this.clock.play() : this.clock.pause();
                    break;
            }
        }
    }

    /** Mirrors SampleApp::handleMouseEvent. */
    handleMouseEvent(mouseEvent: MouseEvent): void {
        this.onMouseEvent(mouseEvent);
    }

    private attachInput(canvas: HTMLCanvasElement): void {
        canvas.tabIndex = 0;
        const on = <K extends keyof HTMLElementEventMap>(target: HTMLElement | Window, type: K, fn: (e: HTMLElementEventMap[K]) => void) => {
            target.addEventListener(type, fn as EventListener);
            this.removeListeners.push(() => target.removeEventListener(type, fn as EventListener));
        };
        on(window, "keydown", (e) => {
            if (document.activeElement instanceof HTMLInputElement) return;
            if (["F2", "F6", "F12", "Space"].includes(e.code)) e.preventDefault();
            this.handleKeyboardEvent(toKeyboardEvent(e, KeyboardEventType.KeyPressed));
        });
        on(window, "keyup", (e) => this.handleKeyboardEvent(toKeyboardEvent(e, KeyboardEventType.KeyReleased)));
        on(canvas, "mousedown", (e) => this.handleMouseEvent(toMouseEvent(e, MouseEventType.ButtonDown, canvas)));
        on(canvas, "mouseup", (e) => this.handleMouseEvent(toMouseEvent(e, MouseEventType.ButtonUp, canvas)));
        on(canvas, "mousemove", (e) => this.handleMouseEvent(toMouseEvent(e, MouseEventType.Move, canvas)));
        on(canvas, "wheel", (e) => this.handleMouseEvent(toMouseEvent(e, MouseEventType.Wheel, canvas)));
        on(canvas, "contextmenu", (e) => e.preventDefault());
        if (this.config.windowDesc?.resizableWindow ?? true) {
            const observer = new ResizeObserver(() => {
                const w = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
                const h = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
                if (w !== canvas.width || h !== canvas.height) this.resizeFrameBuffer(w, h);
            });
            observer.observe(canvas);
            this.removeListeners.push(() => observer.disconnect());
        }
    }
}
