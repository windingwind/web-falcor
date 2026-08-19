/**
 * Pixel inspector mirroring Source/RenderPasses/PixelInspectorPass: reads the
 * G-buffer at a selected pixel, evaluates the material there and exposes the
 * resulting PixelData record. Web divergence (docs §9): the record surfaces on
 * `pixelData` via async readback (~1 frame late) instead of a synchronous
 * getElement; mouse picking is host wiring via setCursorPosition().
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
    ResourceType,
    Texture,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/PixelInspectorPass/PixelInspector.cs.slang";

/** [input name, shader texture name, optional] — mirrors kInputChannels. */
const kInputChannels: [string, string, boolean][] = [
    ["posW", "gWorldPosition", false],
    ["normW", "gWorldShadingNormal", true],
    ["tangentW", "gWorldTangent", true],
    ["faceNormalW", "gWorldFaceNormal", true],
    ["texC", "gTextureCoord", true],
    ["texGrads", "gTextureGrads", true],
    ["mtlData", "gMaterialData", false],
    ["linColor", "gLinearColor", true],
    ["outColor", "gOutputColor", true],
    ["vbuffer", "gVBuffer", true],
];

export const kInvalidIndex = 0xffffffff;

/** Host mirror of PixelInspectorData.slang PixelData. */
export interface PixelData {
    posW: [number, number, number];
    normal: [number, number, number];
    tangent: [number, number, number];
    bitangent: [number, number, number];
    faceNormal: [number, number, number];
    view: [number, number, number];
    texCoord: [number, number];
    frontFacing: number;
    materialID: number;
    doubleSided: number;
    opacity: number;
    IoR: number;
    emission: [number, number, number];
    roughness: number;
    guideNormal: [number, number, number];
    diffuseReflectionAlbedo: [number, number, number];
    diffuseTransmissionAlbedo: [number, number, number];
    specularReflectionAlbedo: [number, number, number];
    specularTransmissionAlbedo: [number, number, number];
    specularReflectance: [number, number, number];
    isTransmissive: number;
    linearColor: [number, number, number, number];
    outputColor: [number, number, number, number];
    luminance: number;
    hitType: number;
    instanceID: number;
    primitiveIndex: number;
    barycentrics: [number, number];
}

// WGSL std430 layout of PixelData (float3 aligns 16, trailing scalars pack into the pad).
const kPixelDataSize = 304;

function decodePixelData(bytes: Uint8Array): PixelData {
    const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const f = (o: number) => v.getFloat32(o, true);
    const u = (o: number) => v.getUint32(o, true);
    const i = (o: number) => v.getInt32(o, true);
    const f2 = (o: number): [number, number] => [f(o), f(o + 4)];
    const f3 = (o: number): [number, number, number] => [f(o), f(o + 4), f(o + 8)];
    const f4 = (o: number): [number, number, number, number] => [f(o), f(o + 4), f(o + 8), f(o + 12)];
    return {
        posW: f3(0),
        normal: f3(16),
        tangent: f3(32),
        bitangent: f3(48),
        faceNormal: f3(64),
        view: f3(80),
        texCoord: f2(96),
        frontFacing: i(104),
        materialID: u(108),
        doubleSided: i(112),
        opacity: f(116),
        IoR: f(120),
        emission: f3(128),
        roughness: f(140),
        guideNormal: f3(144),
        diffuseReflectionAlbedo: f3(160),
        diffuseTransmissionAlbedo: f3(176),
        specularReflectionAlbedo: f3(192),
        specularTransmissionAlbedo: f3(208),
        specularReflectance: f3(224),
        isTransmissive: i(236),
        linearColor: f4(240),
        outputColor: f4(256),
        luminance: f(272),
        hitType: u(276),
        instanceID: u(280),
        primitiveIndex: u(284),
        barycentrics: f2(288),
    };
}

export class PixelInspectorPass extends RenderPass {
    private passes = new Map<string, ComputePass>();
    private pixelDataBuffer: Buffer | null = null;
    private dummyFloat: Texture | null = null;
    private dummyUint: Texture | null = null;
    private cursorPosition: [number, number] = [0, 0];
    private scaleInputsToWindow = false;
    private readbackInFlight = false;
    private warnedDof = false;
    private selectedPixel: [number, number] = [0, 0];

    /** Latest readback (async, ~1 frame late); null until the first record lands. */
    pixelData: PixelData | null = null;

    constructor(device: Device, _props: Properties) {
        super(device);
    }

    /** Mirrors the native mouse picking: normalized [0,1) cursor position. */
    setCursorPosition(x: number, y: number): void {
        this.cursorPosition = [x, y];
    }

    setScaleInputsToWindow(scale: boolean): void {
        this.scaleInputsToWindow = scale;
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        for (const [name, , optional] of kInputChannels) {
            const field = r.addInput(name, `PixelInspector input ${name}`).bindFlags(ResourceBindFlags.ShaderResource);
            if (optional) field.flags(FieldFlags.Optional);
        }
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.passes.clear();
        this.pixelData = null;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        if (!this.scene) return;
        const inputs = new Map<string, Texture | undefined>();
        for (const [name] of kInputChannels) inputs.set(name, renderData.getTexture(name));
        if (!inputs.get("posW") || !inputs.get("mtlData")) return;

        const valid: Record<string, number> = {};
        for (const [name, texname] of kInputChannels) valid[`is_valid_${texname}`] = inputs.get(name) ? 1 : 0;
        const key = JSON.stringify(valid);
        let pass = this.passes.get(key);
        if (!pass) {
            const defines = this.scene.getSceneDefines().addAll(valid);
            pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            this.passes.set(key, pass);
        }
        this.pixelDataBuffer ??= new Buffer(this.device, {
            size: kPixelDataSize,
            structSize: kPixelDataSize,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "PixelInspectorPass::pixelData",
        });

        if (this.scene.camera.getApertureRadius() > 0 && !this.warnedDof) {
            this.warnedDof = true;
            console.warn("PixelInspectorPass assumes a pinhole camera; with DoF enabled the view vector is inaccurate.");
        }

        const [w, h] = renderData.defaultTexDims;
        const px = Math.min(Math.floor(this.cursorPosition[0] * w), w - 1);
        const py = Math.min(Math.floor(this.cursorPosition[1] * h), h - 1);
        this.selectedPixel = [px, py];

        const root = pass.getRootVar();
        this.scene.bindShaderData(root);
        const cb = root["PerFrameCB"]!;
        cb["gResolution"] = [w, h];
        cb["gSelectedPixel"] = [px, py];

        for (const [name, texname] of kInputChannels) {
            const src = inputs.get(name);
            if (src) {
                root[texname] = src;
                // Inputs at a different resolution sample a proportionally scaled coordinate.
                const needsScaling = this.scaleInputsToWindow && (src.width !== w || src.height !== h);
                const scaled = [Math.floor((src.width * px) / w), Math.floor((src.height * py) / h)];
                cb[`${texname}Coord`] = needsScaling ? scaled : [px, py];
            } else {
                // WebGPU has no null bindings: unconnected inputs get 1x1 dummies.
                root[texname] = texname === "gVBuffer" || texname === "gMaterialData" ? this.getDummyUint() : this.getDummyFloat();
                cb[`${texname}Coord`] = [0, 0];
            }
        }
        root["gPixelDataBuffer"] = this.pixelDataBuffer;
        pass.execute(ctx, 1, 1, 1);

        if (!this.readbackInFlight) {
            this.readbackInFlight = true;
            void this.pixelDataBuffer
                .getBlob()
                .then((bytes) => {
                    this.pixelData = decodePixelData(bytes);
                })
                .finally(() => {
                    this.readbackInFlight = false;
                });
        }
    }

    /** Reads the current record directly (test/API convenience). */
    async readPixelData(): Promise<PixelData> {
        if (!this.pixelDataBuffer) throw new Error("PixelInspectorPass: no data — pass has not executed");
        return decodePixelData(await this.pixelDataBuffer.getBlob());
    }

    override renderUI(ui: UIWidgets): void {
        ui.checkbox("Scale inputs to window size", this.scaleInputsToWindow, (v) => (this.scaleInputsToWindow = v));
        const d = this.pixelData;
        if (!d) {
            ui.text("No data yet");
            return;
        }
        const v3 = (v: [number, number, number]) => v.map((x) => x.toFixed(4)).join(", ");
        ui.text(`Looking at pixel (${this.selectedPixel[0]}, ${this.selectedPixel[1]})`);
        const geom = ui.group("Geometry data");
        geom.text(`World position: ${v3(d.posW)}`);
        geom.text(`Shading normal: ${v3(d.normal)}`);
        geom.text(`Tangent: ${v3(d.tangent)}`);
        geom.text(`Face normal: ${v3(d.faceNormal)}`);
        geom.text(`Texture coord: ${d.texCoord.map((x) => x.toFixed(4)).join(", ")}`);
        geom.text(`Front facing: ${d.frontFacing}`);
        const mtl = ui.group("Material data");
        mtl.text(`Material ID: ${d.materialID}`);
        mtl.text(`Double sided: ${d.doubleSided}  IoR: ${d.IoR.toFixed(4)}`);
        mtl.text(`Emission: ${v3(d.emission)}`);
        mtl.text(`Roughness: ${d.roughness.toFixed(4)}`);
        mtl.text(`Diffuse albedo: ${v3(d.diffuseReflectionAlbedo)}`);
        mtl.text(`Specular albedo: ${v3(d.specularReflectionAlbedo)}`);
        const out = ui.group("Output data");
        out.text(`Linear color: ${d.linearColor.map((x) => x.toFixed(4)).join(", ")}`);
        out.text(`Luminance: ${d.luminance.toFixed(4)}`);
        if (d.instanceID !== kInvalidIndex) {
            const vis = ui.group("Visibility data");
            vis.text(`Hit type: ${d.hitType}  instance: ${d.instanceID}  primitive: ${d.primitiveIndex}`);
            vis.text(`Barycentrics: ${d.barycentrics.map((x) => x.toFixed(4)).join(", ")}`);
        }
    }

    private getDummyFloat(): Texture {
        this.dummyFloat ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.RGBA32Float,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "PixelInspectorPass::dummyFloat",
        });
        return this.dummyFloat;
    }

    private getDummyUint(): Texture {
        this.dummyUint ??= new Texture(this.device, {
            type: ResourceType.Texture2D,
            width: 1,
            height: 1,
            format: ResourceFormat.RGBA32Uint,
            bindFlags: ResourceBindFlags.ShaderResource,
            name: "PixelInspectorPass::dummyUint",
        });
        return this.dummyUint;
    }
}

registerRenderPass("PixelInspectorPass", (device, props) => new PixelInspectorPass(device, props));
