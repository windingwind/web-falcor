/**
 * RenderPassReflection field model through RenderGraph allocation
 * (ResourceCache::createResourceForPass parity): raw buffers, 1D/3D/cube
 * textures, full mip chains, format-resolved bind flags, Persistent reuse
 * across recompiles, connected-field merge errors, external inputs in
 * connectedResources.
 */

import {
    Buffer,
    FieldFlags,
    FieldType,
    Properties,
    RenderData,
    RenderGraph,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    Texture,
    kMaxMipLevels,
    type CompileData,
    type Device,
    type RenderContext,
} from "@web-falcor/falcor";
import { gpuTest, expectEq } from "../harness/registry.js";

const SR = ResourceBindFlags.ShaderResource;
const UAV = ResourceBindFlags.UnorderedAccess;
const RT = ResourceBindFlags.RenderTarget;
const DS = ResourceBindFlags.DepthStencil;

/** Declares one field of every type; clears the persistent texture on its first execute. */
class FieldZoo extends RenderPass {
    executions = 0;
    seen = new Map<string, unknown>();
    constructor(device: Device, _props: Properties) {
        super(device);
    }
    override reflect(_cd: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("buf", "raw buffer").rawBuffer(256);
        r.addOutput("tex1d", "1D").texture1D(64).format(ResourceFormat.RGBA8Unorm);
        r.addInternal("cube", "cube map").textureCube(16, 16).format(ResourceFormat.RGBA16Float);
        r.addOutput("vol", "3D").texture3D(8, 8, 4).format(ResourceFormat.R32Float);
        r.addOutput("mips", "full chain").texture2D(64, 64, 1, kMaxMipLevels).format(ResourceFormat.RGBA8Unorm);
        r.addOutput("srgb", "bind flags resolved from format").format(ResourceFormat.RGBA8UnormSrgb);
        r.addOutput("depth", "depth target").format(ResourceFormat.D32Float);
        r.addOutput("hist", "persistent").texture2D(32, 32).format(ResourceFormat.RGBA32Float).flags(FieldFlags.Persistent);
        r.addOutput("scratch", "transient").texture2D(32, 32).format(ResourceFormat.RGBA32Float);
        r.addOutput("color", "for consumers").format(ResourceFormat.RGBA32Float);
        return r;
    }
    override execute(ctx: RenderContext, rd: RenderData): void {
        for (const n of ["buf", "tex1d", "cube", "vol", "mips", "srgb", "depth", "hist", "scratch", "color"]) this.seen.set(n, rd.getResource(n));
        if (this.executions++ === 0) ctx.clearTexture(rd.getTexture("hist")!, [1, 2, 3, 4]);
    }
}

/** Consumer whose input declares a format/size; captures connectedResources. */
class Probe extends RenderPass {
    connected: RenderPassReflection | undefined;
    constructor(device: Device, private readonly inputFormat: ResourceFormat, private readonly inputWidth = 0) {
        super(device);
    }
    override reflect(cd: CompileData): RenderPassReflection {
        this.connected = cd.connectedResources;
        const r = new RenderPassReflection();
        r.addInput("src", "in").texture2D(this.inputWidth, 0).format(this.inputFormat);
        r.addOutput("dst", "out");
        return r;
    }
    override execute(): void {}
}

gpuTest("RenderGraphFields.allocatesEveryFieldType", async ({ device }) => {
    const ctx = device.renderContext;
    const graph = new RenderGraph(device, "Zoo");
    const zoo = graph.addPass(new FieldZoo(device, new Properties()), "Zoo") as FieldZoo;
    graph.markOutput("Zoo.srgb");
    graph.markOutput("Zoo.buf");
    graph.onResize(48, 24);
    graph.execute(ctx);

    const buf = zoo.seen.get("buf") as Buffer;
    expectEq(buf instanceof Buffer, true, "rawBuffer field allocates a Buffer");
    expectEq(buf.size, 256, "buffer size = field width");
    expectEq((buf.bindFlags & (UAV | SR)) === (UAV | SR), true, "raw buffer resolves UAV|SR");
    expectEq(graph.getOutput("Zoo.buf"), undefined, "getOutput hides buffer outputs");
    expectEq(graph.getOutputResource("Zoo.buf") === buf, true, "getOutputResource exposes them");

    const tex1d = zoo.seen.get("tex1d") as Texture;
    expectEq(tex1d.type, ResourceType.Texture1D, "texture1D type");
    expectEq([tex1d.width, tex1d.height], [64, 1], "texture1D dims");
    expectEq(tex1d.bindFlags & (RT | DS), 0, "1D outputs never resolve to render targets (WebGPU rule)");

    const cube = zoo.seen.get("cube") as Texture;
    expectEq(cube.type, ResourceType.TextureCube, "textureCube type");
    expectEq(cube.gpuTexture.depthOrArrayLayers, 6, "cube has 6 faces");

    const vol = zoo.seen.get("vol") as Texture;
    expectEq(vol.type, ResourceType.Texture3D, "texture3D type");
    expectEq(vol.depth, 4, "texture3D depth");

    const mips = zoo.seen.get("mips") as Texture;
    expectEq(mips.mipCount, 7, "kMaxMipLevels -> full chain for 64x64");

    const srgb = zoo.seen.get("srgb") as Texture;
    expectEq([srgb.width, srgb.height], [48, 24], "size-0 field takes the graph dims");
    expectEq(srgb.bindFlags & UAV, 0, "sRGB output gets no UAV (not storage-capable)");
    expectEq(srgb.bindFlags & (SR | RT), SR | RT, "sRGB output resolves SR|RT");

    const depth = zoo.seen.get("depth") as Texture;
    expectEq(depth.bindFlags, SR | DS, "depth output resolves SR|DS only");

    const color = zoo.seen.get("color") as Texture;
    expectEq(color.bindFlags, SR | UAV | RT, "float output resolves SR|UAV|RT");
});

gpuTest("RenderGraphFields.persistentSurvivesRecompile", async ({ device }) => {
    const ctx = device.renderContext;
    const graph = new RenderGraph(device, "Zoo");
    const zoo = graph.addPass(new FieldZoo(device, new Properties()), "Zoo") as FieldZoo;
    graph.onResize(32, 32);
    graph.execute(ctx);
    const hist0 = zoo.seen.get("hist") as Texture;
    const scratch0 = zoo.seen.get("scratch") as Texture;

    graph.onResize(32, 32); // same dims: forces a recompile without changing any field
    graph.execute(ctx);
    expectEq(zoo.seen.get("hist") === hist0, true, "Persistent field keeps its texture across recompiles");
    expectEq(zoo.seen.get("scratch") !== scratch0, true, "transient field is reallocated");
    const px = new Float32Array((await ctx.readTextureSubresource(hist0)).buffer);
    expectEq([px[0], px[1], px[2], px[3]], [1, 2, 3, 4], "persistent contents survive (cleared only on the first execute)");

    graph.onResize(64, 64); // dims change: size-0 fields differ, fixed-size persistent field still matches
    graph.execute(ctx);
    expectEq(zoo.seen.get("hist") === hist0, true, "fixed-size persistent field survives a resize");
});

gpuTest("RenderGraphFields.mergeMismatchThrowsAndConnectedShapesPropagate", async ({ device }) => {
    const ctx = device.renderContext;
    {
        const graph = new RenderGraph(device, "Merge");
        graph.addPass(new FieldZoo(device, new Properties()), "Zoo");
        graph.addPass(new Probe(device, ResourceFormat.RGBA16Float), "Probe");
        graph.addEdge("Zoo.color", "Probe.src");
        let threw = "";
        try {
            graph.compile(ctx);
        } catch (e) {
            threw = String(e);
        }
        expectEq(/Format already specified/.test(threw), true, `conflicting connected formats throw (got: ${threw})`);
    }
    {
        const graph = new RenderGraph(device, "Connected");
        graph.addPass(new FieldZoo(device, new Properties()), "Zoo");
        const probe = graph.addPass(new Probe(device, ResourceFormat.Unknown), "Probe") as Probe;
        graph.addEdge("Zoo.color", "Probe.src");
        graph.onResize(20, 10);
        graph.compile(ctx);
        const f = probe.connected!.getField("src")!;
        expectEq([f.type_, f.format_], [FieldType.Texture2D, ResourceFormat.RGBA32Float], "connected source shape visible to the consumer's reflect");
        const bound = graph.getOutput("Zoo.color")!;
        expectEq([bound.width, bound.height], [20, 10], "merged output allocated at graph dims");
    }
    {
        const graph = new RenderGraph(device, "External");
        const probe = graph.addPass(new Probe(device, ResourceFormat.Unknown), "Probe") as Probe;
        const ext = new Texture(device, { type: ResourceType.Texture2D, width: 12, height: 6, format: ResourceFormat.RG32Float, bindFlags: SR, mipLevels: 1 });
        graph.setInput("Probe.src", ext);
        graph.compile(ctx);
        const f = probe.connected!.getField("src")!;
        expectEq([f.width, f.height, f.format_], [12, 6, ResourceFormat.RG32Float], "external input reflected into connectedResources");
    }
});

gpuTest("RenderGraphFields.requestRecompileRecompilesNextFrame", async ({ device }) => {
    const ctx = device.renderContext;
    const graph = new RenderGraph(device, "Recompile");
    const zoo = graph.addPass(new FieldZoo(device, new Properties()), "Zoo") as FieldZoo;
    graph.onResize(16, 16);
    graph.execute(ctx);
    const scratch0 = zoo.seen.get("scratch");
    graph.execute(ctx);
    expectEq(zoo.seen.get("scratch") === scratch0, true, "no recompile without a request");
    zoo.requestRecompile();
    graph.execute(ctx);
    expectEq(zoo.seen.get("scratch") !== scratch0, true, "requestRecompile() recompiles before the next execute");
    expectEq(zoo.recompileRequested, false, "request flag consumed");
});

/** Output with caller-chosen bind flags plus an optional input that only asks for ShaderResource. */
class FlagPair extends RenderPass {
    constructor(device: Device, private readonly outFlags: ResourceBindFlags) {
        super(device);
    }
    override reflect(): RenderPassReflection {
        const r = new RenderPassReflection();
        r.addOutput("out", "producer").format(ResourceFormat.RGBA32Float).bindFlags(this.outFlags);
        r.addInput("src", "SR-only consumer").bindFlags(SR).flags(FieldFlags.Optional);
        return r;
    }
    override execute(): void {}
}

gpuTest("RenderGraphFields.srOnlyConsumerKeepsProducerWritable", async ({ device }) => {
    const ctx = device.renderContext;
    const flagsFor = (outFlags: ResourceBindFlags) => {
        const graph = new RenderGraph(device, "SrOnly");
        graph.addPass(new FlagPair(device, outFlags), "P");
        graph.addPass(new FlagPair(device, outFlags), "C");
        graph.addEdge("P.out", "C.src");
        graph.onResize(8, 8);
        graph.compile(ctx);
        return graph.getOutput("P.out")!.bindFlags;
    };
    // ResourceCache::registerField decides resolution per alias before merging, so a None
    // output stays writable even when every consumer only asks for ShaderResource.
    expectEq(flagsFor(ResourceBindFlags.None), SR | UAV | RT, "None output + SR-only consumer resolves SR|UAV|RT");
    expectEq(flagsFor(UAV), UAV | SR, "explicit UAV output + SR consumer merges to UAV|SR, nothing resolved");
});
