/**
 * Port of Falcor/Source/RenderPasses/TestPasses/TestRtProgram.
 *
 * Native exercises shader-table plumbing: two ray types and two miss shaders
 * (mode 0, with custom-primitive intersection shaders) or type-conformance
 * specialized hit groups (mode 1). The web lowers the binding table to a
 * compute kernel over the software BVH (docs §5); the per-pixel colours match.
 */
import {
    Buffer,
    canonicalFloat,
    ComputePass,
    Logger,
    MemoryType,
    Mt19937,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    RuntimeError,
    registerRenderPass,
    type CompileData,
    type Device,
    type RenderContext,
    type UIWidgets,
} from "@web-falcor/falcor";

const kShaderFile = "RenderPasses/TestPasses/TestRtProgram.rt.slang";
const kMode = "mode";
const kOutput = "output";

// Native keeps one file-scope std::mt19937 shared by all TestRtProgram instances.
const rng = new Mt19937();

type Aabb = { min: [number, number, number]; max: [number, number, number] };

export class TestRtProgram extends RenderPass {
    private mode = 0;
    private pass: ComputePass | null = null;
    private geometryIDs: Buffer | null = null;
    private userID = 0;
    private selectedIdx = 0;
    private prevSelectedIdx = -1;
    private selectedAABB: Aabb = { min: [0, 0, 0], max: [0, 0, 0] };

    constructor(device: Device, props: Properties) {
        super(device);
        for (const [key, value] of props.entries()) {
            if (key === kMode) this.mode = Number(value);
            else Logger.warning(`Unknown property '${key}' in TestRtProgram properties.`);
        }
        if (this.mode !== 0 && this.mode !== 1) throw new RuntimeError("mode has to be 0 or 1");
    }

    override getProperties(): Properties {
        return new Properties({ [kMode]: this.mode });
    }

    override reflect(_compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput(kOutput, "Output image").bindFlags(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.RenderTarget).format(ResourceFormat.RGBA32Float);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.pass = null;
        this.geometryIDs = null;
    }

    private makeBuffer(data: Uint32Array | Float32Array, name: string): Buffer {
        const buf = new Buffer(this.device, {
            size: data.byteLength,
            structSize: data instanceof Float32Array ? 16 : 4,
            bindFlags: ResourceBindFlags.ShaderResource,
            memoryType: MemoryType.DeviceLocal,
            name,
        });
        buf.setBlob(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        return buf;
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const output = renderData.getTexture(kOutput)!;
        ctx.clearTexture(output, [0, 0, 0, 1]);
        if (!this.scene) return;
        const scene = this.scene;

        if (!this.pass) {
            const defines = scene.getSceneDefines().addAll({ MODE: this.mode });
            this.pass = ComputePass.create(this.device, { path: kShaderFile, defines });
            const ids = scene.getMeshIDs();
            this.geometryIDs = this.makeBuffer(ids.length > 0 ? ids : new Uint32Array(1), "TestRtProgram::geometryIDs");
        }

        // Native rebuilds the AABB BLAS on custom-primitive edits; re-upload the few AABBs per frame.
        const count = scene.getCustomPrimitiveCount();
        const prims = new Float32Array(Math.max(1, count) * 8);
        const bits = new Uint32Array(prims.buffer);
        for (let i = 0; i < count; i++) {
            const aabb = scene.getCustomPrimitiveAABB(i);
            prims.set([...aabb.min, 0, ...aabb.max, 0], i * 8);
            bits[i * 8 + 3] = scene.getCustomPrimitive(i).userID;
        }

        const root = this.pass.getRootVar();
        scene.bindShaderData(root);
        root["gTestProgram"]["frameDim"] = [output.width, output.height];
        root["gOutput"] = output;
        root["gGeometryIDs"] = this.geometryIDs!;
        root["gCustomPrimitives"] = this.makeBuffer(prims, "TestRtProgram::customPrimitives");
        root["CB"]["gCustomPrimitiveCount"] = count;
        this.pass.execute(ctx, output.width, output.height);
    }

    override renderUI(ui: UIWidgets): void {
        if (!this.scene) {
            ui.text("No scene loaded!");
            return;
        }
        ui.text(`Test mode: ${this.mode}`);
        if (this.mode !== 0) return;

        const primCount = this.scene.getCustomPrimitiveCount();
        ui.text(`Custom primitives: ${primCount}`);
        this.selectedIdx = Math.max(0, Math.min(this.selectedIdx, primCount - 1));
        ui.text("\nSelected primitive:");
        ui.slider("##idx", this.selectedIdx, 0, Math.max(0, primCount - 1), 1, (v) => (this.selectedIdx = Math.round(v)));
        if (primCount > 0 && this.selectedIdx !== this.prevSelectedIdx) {
            this.prevSelectedIdx = this.selectedIdx;
            this.selectedAABB = this.scene.getCustomPrimitiveAABB(this.selectedIdx);
        }
        ui.button("Add", () => this.addCustomPrimitive());
        if (primCount > 0) {
            ui.button("Remove", () => this.removeCustomPrimitive(this.selectedIdx));
            ui.button("Random move", () => this.moveCustomPrimitive());
            // No float3 widget: one slider per component.
            for (const [label, v] of [["Min", this.selectedAABB.min], ["Max", this.selectedAABB.max]] as const) {
                v.forEach((x, i) => ui.slider(`${label}.${"xyz"[i]}`, x, -100, 100, 0.01, (nv) => (v[i] = nv)));
            }
            ui.button("Update", () => this.scene?.updateCustomPrimitive(this.selectedIdx, this.selectedAABB));
        }
    }

    /** Adds a random sphere-bounding AABB (native draws from the shared mt19937). */
    addCustomPrimitive(): void {
        if (!this.scene) {
            Logger.warning("No scene! Ignoring call to addCustomPrimitive()");
            return;
        }
        const u = () => canonicalFloat(rng);
        const fr = Math.fround;
        // Braced-init order: x, y, z draws are sequenced left to right.
        const c = [fr(4 * u() - 2), u(), fr(4 * u() - 2)];
        const r = fr(0.5 * u() + 0.5);
        this.scene.addCustomPrimitive(this.userID++, { min: c.map((x) => fr(x - r)) as Aabb["min"], max: c.map((x) => fr(x + r)) as Aabb["max"] });
    }

    removeCustomPrimitive(index: number): void {
        if (!this.scene) {
            Logger.warning("No scene! Ignoring call to removeCustomPrimitive()");
            return;
        }
        if (index >= this.scene.getCustomPrimitiveCount()) {
            Logger.warning("Custom primitive index is out of range. Ignoring call to removeCustomPrimitive()");
            return;
        }
        this.scene.removeCustomPrimitives(index, index + 1);
    }

    moveCustomPrimitive(): void {
        if (!this.scene) {
            Logger.warning("No scene! Ignoring call to moveCustomPrimitive()");
            return;
        }
        const primCount = this.scene.getCustomPrimitiveCount();
        if (primCount === 0) {
            Logger.warning("Scene has no custom primitives. Ignoring call to moveCustomPrimitive()");
            return;
        }
        const u = () => canonicalFloat(rng);
        const fr = Math.fround;
        const index = Math.min(Math.floor(fr(u() * primCount)), primCount - 1);
        const aabb = this.scene.getCustomPrimitiveAABB(index);
        const d = [u(), u(), u()].map((x) => fr(fr(x * 2) - 1));
        this.scene.updateCustomPrimitive(index, {
            min: aabb.min.map((x, i) => fr(x + d[i]!)) as Aabb["min"],
            max: aabb.max.map((x, i) => fr(x + d[i]!)) as Aabb["max"],
        });
    }
}

registerRenderPass("TestRtProgram", (device, props) => new TestRtProgram(device, props));
