/**
 * Mirrors Samples/HelloDXR: loads a scene and draws it either rasterized (direct analytic
 * lighting) or ray traced (plus shadows and a one-bounce reflection); Space toggles.
 * §9: rasterization pulls vertices from the scene buffers (WGSL has no SV_PrimitiveID)
 * and the RT program runs as a compute kernel over the software BVH, so the shader table
 * has no web counterpart. Native Scene forwards input to its first-person camera
 * controller; here the sample owns that controller.
 */

import {
    ComputePass,
    CullMode,
    DepthStencilState,
    DepthStencilStateDesc,
    FboAttachmentType,
    FirstPersonCameraController,
    GraphicsState,
    InputClass,
    KeyboardEventType,
    MouseButton,
    ParameterBlock,
    RasterizerState,
    RasterizerStateDesc,
    ResourceBindFlags,
    ResourceFormat,
    SampleApp,
    ShaderType,
    Topology,
    Vao,
    VertexBufferLayout,
    VertexLayout,
    float2,
    focalLengthToFovY,
    inverse,
    makeRootVar,
    mergeWgslBindings,
    initScripting,
    isScriptingInitialized,
    runSceneScript,
    type Fbo,
    type KeyboardEvent,
    type MouseEvent,
    type RenderContext,
    type SampleAppConfig,
    type Scene,
    type ShaderVar,
    type Texture,
    type UIWidgets,
} from "@web-falcor/falcor";

const kClearColor: [number, number, number, number] = [0.38, 0.52, 0.1, 1];
export const kDefaultScene = "/Falcor/media/Arcade/Arcade.pyscene";

interface RasterProgram {
    state: GraphicsState;
    vars: ParameterBlock;
    root: ShaderVar;
    pipelineLayout: GPUPipelineLayout;
}

export class HelloDXR extends SampleApp {
    static readonly config: SampleAppConfig = { windowDesc: { title: "HelloDXR", resizableWindow: true } };
    scene: Scene | null = null;
    rayTrace = true;
    useDOF = false;
    private raster: RasterProgram | null = null;
    private raytracePass: ComputePass | null = null;
    private rtOut: Texture | null = null;
    private sampleIndex = 0;
    private camControl: FirstPersonCameraController | null = null;

    override async onLoad(): Promise<void> {
        await this.loadScene(kDefaultScene);
    }

    override onResize(width: number, height: number): void {
        const camera = this.scene?.camera;
        if (camera) {
            camera.setFocalLength(18);
            camera.setAspectRatio(width / height);
        }
        this.rtOut = this.getDevice().createTexture2D(width, height, ResourceFormat.RGBA16Float, 1, 1, undefined, ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource | ResourceBindFlags.RenderTarget);
    }

    override onFrameRender(renderContext: RenderContext, targetFbo: Fbo): void {
        renderContext.clearFbo(targetFbo, kClearColor, 1.0, 0, FboAttachmentType.All);
        if (this.scene) {
            this.camControl?.update(performance.now() / 1000);
            if (this.scene.isAnimated()) this.scene.animate(this.getGlobalClock().getTime());
            if (this.rayTrace) this.renderRT(renderContext, targetFbo);
            else this.renderRaster(renderContext, targetFbo);
        }
        this.getTextRenderer().render(renderContext, this.getFrameRate().getMsg(), targetFbo, [20, 20]);
    }

    override onGuiRender(gui: UIWidgets): void {
        const w = gui.group("Hello DXR Settings");
        w.checkbox("Ray Trace", this.rayTrace, (v) => (this.rayTrace = v));
        w.checkbox("Use Depth of Field", this.useDOF, (v) => (this.useDOF = v));
        this.renderGlobalUI(w);
    }

    override onKeyEvent(keyEvent: KeyboardEvent): boolean {
        if (keyEvent.key === "Space" && keyEvent.type === KeyboardEventType.KeyPressed) {
            this.rayTrace = !this.rayTrace;
            return true;
        }
        if (keyEvent.type !== KeyboardEventType.KeyPressed && keyEvent.type !== KeyboardEventType.KeyReleased) return false;
        return (
            this.camControl?.onKeyEvent({ type: keyEvent.type === KeyboardEventType.KeyPressed ? "keyPressed" : "keyReleased", key: keyEvent.key.toLowerCase(), shift: (keyEvent.mods & 1) !== 0, ctrl: (keyEvent.mods & 2) !== 0 }) ?? false
        );
    }

    override onMouseEvent(mouseEvent: MouseEvent): boolean {
        const type = (["buttonDown", "buttonUp", "move", "wheel"] as const)[mouseEvent.type];
        const button = mouseEvent.button === MouseButton.Left ? "left" : mouseEvent.button === MouseButton.Right ? "right" : "middle";
        return this.camControl?.onMouseEvent({ type, button, pos: new float2(mouseEvent.pos[0], mouseEvent.pos[1]), wheelDelta: new float2(mouseEvent.wheelDelta[0], mouseEvent.wheelDelta[1]) }) ?? false;
    }

    /** Mirrors HelloDXR::loadScene: scene, camera depth range/speed, raster and RT programs. */
    async loadScene(url: string): Promise<void> {
        const device = this.getDevice();
        // .pyscene files run through Pyodide.
        if (!isScriptingInitialized()) await initScripting("/node_modules/pyodide");
        const baseUrl = url.slice(0, url.lastIndexOf("/"));
        this.scene = await runSceneScript(device, await (await fetch(url)).text(), baseUrl);
        const camera = this.scene.camera;

        // Update the controllers.
        const b = this.scene.worldBounds;
        const radius = b ? 0.5 * Math.hypot(b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]) : 1;
        this.camControl = new FirstPersonCameraController(camera);
        this.camControl.setCameraSpeed(radius * 0.25);
        const nearZ = Math.max(0.1, radius / 750);
        const farZ = radius * 10;
        camera.setDepthRange(nearZ, farZ);
        const fbo = this.getTargetFbo();
        camera.setAspectRatio(fbo.width / fbo.height);

        const defines = this.scene.getSceneDefines();
        this.raster = this.createRasterProgram(defines);
        this.raytracePass = ComputePass.create(device, { path: "Samples/HelloDXR/HelloDXR.rt.slang", csEntry: "main", defines });
    }

    private createRasterProgram(defines: ReturnType<Scene["getSceneDefines"]>): RasterProgram {
        const device = this.getDevice();
        const program = device.programManager.createProgram(
            {
                path: "Samples/HelloDXR/HelloDXR.3d.slang",
                entryPoints: [
                    { name: "vsMain", type: ShaderType.Vertex },
                    { name: "psMain", type: ShaderType.Pixel },
                ],
            },
            defines,
        );
        const version = program.getActiveVersion();
        const vs = version.getKernel("vsMain", ShaderType.Vertex);
        const ps = version.getKernel("psMain", ShaderType.Pixel);
        const vars = new ParameterBlock(device, version.reflection, mergeWgslBindings(vs.bindings, ps.bindings));
        const state = new GraphicsState(device).setKernels(vs, ps);
        // Vertex pulling: only the per-instance draw-ID stream (geometry comes from the scene buffers).
        const ib = new VertexBufferLayout().addElement("DRAW_ID", 0, ResourceFormat.R32Uint, 1, 0).setInputClass(InputClass.PerInstanceData, 1);
        state.setVao(new Vao(Topology.TriangleList, new VertexLayout().addBufferLayout(0, ib), [this.scene!.getMeshDrawData().drawIDBuffer]));
        // Native Scene::rasterize culls back faces per mesh winding; the web draws both sides (like GBufferRaster).
        state.setRasterizerState(RasterizerState.create(new RasterizerStateDesc().setCullMode(CullMode.None)));
        state.setDepthStencilState(DepthStencilState.create(new DepthStencilStateDesc()));
        const groups = vars.getGroupIndices();
        const layouts: GPUBindGroupLayout[] = [];
        for (let g = 0; g <= (groups.length ? Math.max(...groups) : -1); g++) layouts.push(vars.getBindGroupLayout(g) ?? device.gpuDevice.createBindGroupLayout({ entries: [] }));
        return { state, vars, root: makeRootVar(vars), pipelineLayout: device.gpuDevice.createPipelineLayout({ bindGroupLayouts: layouts }) };
    }

    private renderRaster(ctx: RenderContext, targetFbo: Fbo): void {
        const scene = this.scene!;
        const raster = this.raster!;
        scene.bindShaderData(raster.root);
        raster.state.setFbo(targetFbo);
        const gso = raster.state.getGSO(raster.pipelineLayout);
        const bindGroups = raster.vars.getGroupIndices().map((g) => ({ index: g, group: raster.vars.getBindGroup(g) }));
        const pass = ctx.getEncoder().beginRenderPass(targetFbo.getGpuRenderPassDescriptor());
        pass.setPipeline(gso.gpuPipeline);
        pass.setViewport(0, 0, targetFbo.width, targetFbo.height, 0, 1);
        for (const { index, group } of bindGroups) pass.setBindGroup(index, group);
        raster.state.getVao()!.vertexBuffers.forEach((vb, i) => pass.setVertexBuffer(i, vb.gpuBuffer));
        for (const draw of scene.getMeshDrawData().draws) pass.draw(draw.indexCount, 1, 0, draw.firstInstance);
        pass.end();
    }

    private setPerFrameVars(targetFbo: Fbo): void {
        const camera = this.scene!.camera;
        const root = this.raytracePass!.getRootVar();
        const cb = root["PerFrameCB"];
        cb["invView"] = inverse(camera.getViewMatrix());
        cb["viewportDims"] = [targetFbo.width, targetFbo.height];
        const fovY = focalLengthToFovY(camera.getFocalLength(), 24);
        cb["tanHalfFovY"] = Math.tan(fovY * 0.5);
        cb["sampleIndex"] = this.sampleIndex++;
        cb["useDOF"] = this.useDOF;
        root["gOutput"] = this.rtOut!;
    }

    private renderRT(ctx: RenderContext, targetFbo: Fbo): void {
        this.setPerFrameVars(targetFbo);
        ctx.clearTexture(this.rtOut!, kClearColor);
        this.scene!.bindShaderData(this.raytracePass!.getRootVar());
        this.raytracePass!.execute(ctx, targetFbo.width, targetFbo.height);
        ctx.blit(this.rtOut!, targetFbo.getColorTexture(0)!);
    }
}
