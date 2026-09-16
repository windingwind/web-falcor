import { describe, expect, it } from "vitest";
import { Field, FieldFlags, FieldType, FieldVisibility, RenderPassReflection, kMaxMipLevels } from "../src/RenderGraph/RenderPassReflection.js";
import { ResourceFormat, getFormatBindFlags } from "../src/Core/API/Formats.js";
import { ResourceBindFlags, ResourceType } from "../src/Core/API/Types.js";

describe("RenderPassReflection.Field builders (native RenderPassReflection.cpp semantics)", () => {
    it("rawBuffer stores the byte size in width and zeroes texture dims", () => {
        const f = new Field("buf", "", FieldVisibility.Output).rawBuffer(1024);
        expect(f.type_).toBe(FieldType.RawBuffer);
        expect([f.width, f.height, f.depth, f.arraySize, f.mipCount]).toEqual([1024, 0, 0, 0, 0]);
        expect(f.getResourceType()).toBe(ResourceType.Buffer);
    });

    it("texture1D/2D/3D/Cube set the implied dimensions", () => {
        const t1 = new Field().texture1D(64, 3, 2);
        expect([t1.type_, t1.width, t1.height, t1.depth, t1.sampleCount, t1.mipCount, t1.arraySize]).toEqual([FieldType.Texture1D, 64, 1, 1, 1, 3, 2]);
        const t2 = new Field().texture2D(8, 4, 4);
        expect([t2.type_, t2.depth, t2.sampleCount]).toEqual([FieldType.Texture2D, 1, 4]);
        expect(t2.getResourceType()).toBe(ResourceType.Texture2DMultisample);
        const t3 = new Field().texture3D(8, 8, 2, 3);
        expect([t3.type_, t3.depth, t3.arraySize, t3.mipCount]).toEqual([FieldType.Texture3D, 2, 3, 1]);
        const tc = new Field().textureCube(16, 16, kMaxMipLevels, 2);
        expect([tc.type_, tc.depth, tc.mipCount, tc.arraySize]).toEqual([FieldType.TextureCube, 1, kMaxMipLevels, 2]);
        expect(tc.getResourceType()).toBe(ResourceType.TextureCube);
    });

    it("resourceType() dispatches to the typed builder", () => {
        const f = new Field().resourceType(FieldType.Texture3D, 4, 5, 6, 1, 1, 1);
        expect([f.type_, f.width, f.height, f.depth]).toEqual([FieldType.Texture3D, 4, 5, 6]);
        const b = new Field().resourceType(FieldType.RawBuffer, 256, 0, 0, 0, 0, 0);
        expect([b.type_, b.width]).toEqual([FieldType.RawBuffer, 256]);
    });

    it("isValid rejects multisampled mip chains and optional internals", () => {
        expect(new Field("a", "", FieldVisibility.Output).texture2D(0, 0, 4, 2).isValid()).toBe(false);
        expect(new Field("b", "", FieldVisibility.Internal).flags(FieldFlags.Optional).isValid()).toBe(false);
        expect(new Field("c", "", FieldVisibility.Output).texture2D(0, 0, 4, 1).isValid()).toBe(true);
    });

    it("flags/visibility predicates", () => {
        const f = new Field("f", "", FieldVisibility.Input | FieldVisibility.Output).flags(FieldFlags.Persistent | FieldFlags.Optional);
        expect([f.isInput(), f.isOutput(), f.isInternal(), f.isOptional(), f.isPersistent()]).toEqual([true, true, false, true, true]);
    });
});

describe("Field.merge / equals", () => {
    it("fills unspecified properties from the other field and ORs visibility/bind flags", () => {
        const out = new Field("dst", "", FieldVisibility.Output).texture2D(0, 0).bindFlags(ResourceBindFlags.UnorderedAccess);
        const inp = new Field("src", "", FieldVisibility.Input).texture2D(128, 64).format(ResourceFormat.RGBA16Float).bindFlags(ResourceBindFlags.ShaderResource);
        out.merge(inp);
        expect([out.width, out.height, out.format_]).toEqual([128, 64, ResourceFormat.RGBA16Float]);
        expect(out.visibility_).toBe(FieldVisibility.Input | FieldVisibility.Output);
        expect(out.bindFlags_).toBe(ResourceBindFlags.UnorderedAccess | ResourceBindFlags.ShaderResource);
    });

    it("keeps the base value when the other side is unspecified", () => {
        const out = new Field("dst", "", FieldVisibility.Output).texture2D(32, 32).format(ResourceFormat.R32Float);
        out.merge(new Field("src", "", FieldVisibility.Input).texture2D(0, 0));
        expect([out.width, out.format_]).toEqual([32, ResourceFormat.R32Float]);
    });

    it("throws on conflicting dimensions, formats and types", () => {
        const base = () => new Field("dst", "", FieldVisibility.Output).texture2D(32, 32).format(ResourceFormat.R32Float);
        expect(() => base().merge(new Field("a", "", FieldVisibility.Input).texture2D(64, 32))).toThrow(/Width already specified/);
        expect(() => base().merge(new Field("b", "", FieldVisibility.Input).format(ResourceFormat.RGBA8Unorm))).toThrow(/Format already specified/);
        expect(() => base().merge(new Field("c", "", FieldVisibility.Input).texture3D(0, 0, 0))).toThrow(/mismatching types/);
        expect(() => base().merge(new Field("d", "", FieldVisibility.Internal))).toThrow(/internal/);
    });

    it("equals compares every property; clone round-trips", () => {
        const a = new Field("x", "desc", FieldVisibility.Output).texture2D(4, 4, 1, 2, 3).format(ResourceFormat.RG32Float).flags(FieldFlags.Persistent);
        expect(a.clone().equals(a)).toBe(true);
        expect(a.clone().flags(FieldFlags.None).equals(a)).toBe(false);
        expect(a.clone().desc("other").equals(a)).toBe(false);
    });
});

describe("RenderPassReflection field set", () => {
    it("addInput/addOutput leave bind flags to allocation-time resolution (None)", () => {
        const r = new RenderPassReflection();
        expect(r.addInput("in", "").bindFlags_).toBe(ResourceBindFlags.None);
        expect(r.addOutput("out", "").bindFlags_).toBe(ResourceBindFlags.None);
        expect(r.getFieldCount()).toBe(2);
        expect(r.getField(1)?.name_).toBe("out");
    });

    it("re-adding a name merges I/O visibility instead of duplicating", () => {
        const r = new RenderPassReflection();
        r.addInput("io", "in");
        const f = r.addOutput("io", "out");
        expect(r.getFieldCount()).toBe(1);
        expect(f.visibility_).toBe(FieldVisibility.Input | FieldVisibility.Output);
        expect(r.addInternal("io", "x")).toBe(f); // mismatching request: warns, keeps the existing field
        expect(r.getFieldCount()).toBe(1);
    });

    it("equals is order-independent", () => {
        const a = new RenderPassReflection();
        a.addInput("i", "").texture2D(0, 0);
        a.addOutput("o", "").format(ResourceFormat.RGBA32Float);
        const b = new RenderPassReflection();
        b.addOutput("o", "").format(ResourceFormat.RGBA32Float);
        b.addInput("i", "").texture2D(0, 0);
        expect(a.equals(b)).toBe(true);
        b.getField("o")!.format(ResourceFormat.RGBA16Float);
        expect(a.equals(b)).toBe(false);
    });

    it("addConnectedField clones a source under the input's name", () => {
        const src = new Field("dst", "d", FieldVisibility.Output).texture2D(8, 8).format(ResourceFormat.R32Float);
        const r = new RenderPassReflection();
        const f = r.addConnectedField("src", src);
        expect([f.name_, f.width, f.format_]).toEqual(["src", 8, ResourceFormat.R32Float]);
        expect(f).not.toBe(src);
    });
});

describe("getFormatBindFlags (WebGPU capability table)", () => {
    const SR = ResourceBindFlags.ShaderResource, UAV = ResourceBindFlags.UnorderedAccess, RT = ResourceBindFlags.RenderTarget, DS = ResourceBindFlags.DepthStencil;
    it("storage-capable color formats get SR|UAV|RT; sRGB/16-bit two-channel get SR|RT", () => {
        expect(getFormatBindFlags(ResourceFormat.RGBA32Float)).toBe(SR | UAV | RT);
        expect(getFormatBindFlags(ResourceFormat.R32Uint)).toBe(SR | UAV | RT);
        expect(getFormatBindFlags(ResourceFormat.RGBA8UnormSrgb)).toBe(SR | RT);
        expect(getFormatBindFlags(ResourceFormat.RG16Float)).toBe(SR | RT);
    });
    it("depth, compressed, non-renderable and feature-gated formats", () => {
        expect(getFormatBindFlags(ResourceFormat.D32Float)).toBe(SR | DS);
        expect(getFormatBindFlags(ResourceFormat.BC7Unorm)).toBe(SR);
        expect(getFormatBindFlags(ResourceFormat.RGB9E5Float)).toBe(SR);
        expect(getFormatBindFlags(ResourceFormat.BGRA8Unorm)).toBe(SR | RT);
        expect(getFormatBindFlags(ResourceFormat.BGRA8Unorm, (f) => f === "bgra8unorm-storage")).toBe(SR | RT | UAV);
        expect(getFormatBindFlags(ResourceFormat.Unknown)).toBe(ResourceBindFlags.None);
        expect(getFormatBindFlags(ResourceFormat.RGB32Float)).toBe(ResourceBindFlags.None);
    });
});
