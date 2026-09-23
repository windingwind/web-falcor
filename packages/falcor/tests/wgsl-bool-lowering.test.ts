import { describe, expect, it } from "vitest";
import { lowerHostShareableBools, lowerWriteOnlyStorageTextures } from "../src/Core/Program/WgslBoolLowering.js";

const kSource = `struct S2_std140_0
{
    @align(16) a_1 : vec3<bool>,
    @align(16) b_1 : f32,
};
struct S3_std140_0
{
    @align(16) a_2 : u32,
    @align(4) b_2 : bool,
    @align(16) s2_0 : S2_std140_0,
};
struct Local_0
{
    x_0 : bool,
};
@binding(1) @group(0) var<uniform> CB_0 : S3_std140_0;
@binding(2) @group(0) var<storage, read_write> buf_0 : array<S3_std140_0>;
fn main()
{
    if(CB_0.b_2)
    {
    }
    var l : Local_0;
    l.x_0 = CB_0.s2_0.a_1.y;
    var v : vec3<bool> = CB_0.s2_0.a_1;
    buf_0[i32(3)].b_2 = !l.x_0 && CB_0.b_2;
}`;

describe("lowerHostShareableBools", () => {
    const out = lowerHostShareableBools(kSource);
    it("retypes bools in layout structs only", () => {
        expect(out).toContain("@align(4) b_2 : u32,");
        expect(out).toContain("@align(16) a_1 : vec3<u32>,");
        expect(out).toContain("x_0 : bool,");
    });
    it("converts reads and writes", () => {
        expect(out).toContain("if((CB_0.b_2 != 0u))");
        expect(out).toContain("l.x_0 = (CB_0.s2_0.a_1.y != 0u);");
        expect(out).toContain("var v : vec3<bool> = (CB_0.s2_0.a_1 != vec3<u32>(0u));");
        expect(out).toContain("buf_0[i32(3)].b_2 = select(0u, 1u, !l.x_0 && (CB_0.b_2 != 0u));");
    });
    it("leaves modules without layout bools untouched", () => {
        const plain = "struct A_std140_0 { @align(4) f_0 : f32, };\nfn main() {}";
        expect(lowerHostShareableBools(plain)).toBe(plain);
    });
});

describe("relaxSubgroupUniformity", () => {
    it("adds the diagnostic after enable directives when subgroup builtins are used", async () => {
        const { relaxSubgroupUniformity } = await import("../src/Core/Program/WgslBoolLowering.js");
        const src = "enable subgroups;\nfn f(v : f32) -> f32 { return subgroupMax(v); }";
        expect(relaxSubgroupUniformity(src)).toBe("enable subgroups;\ndiagnostic(off, subgroup_uniformity);\nfn f(v : f32) -> f32 { return subgroupMax(v); }");
        expect(relaxSubgroupUniformity("fn f() {}")).toBe("fn f() {}");
    });
});

describe("lowerHostShareableBools: pointer and call chains", () => {
    it("wraps accesses through a parenthesized deref", () => {
        const src = "struct T_std140_0\n{\n    @align(4) v_0 : bool,\n};\nfn f(p : ptr<function, T_std140_0>) { if((*p).v_0) { } }";
        expect(lowerHostShareableBools(src)).toContain("if(((*p).v_0 != 0u))");
    });
});

describe("lowerWriteOnlyStorageTextures", () => {
    const decl = (name: string, format: string) => `@binding(0) @group(0) var ${name} : texture_storage_2d<${format}, read_write>;`;
    it("makes unread non-r32 storage textures write-only", () => {
        const wgsl = `${decl("g_output_0", "rgba32float")}\nfn main() { textureStore(g_output_0, vec2<u32>(0), vec4<f32>(1)); }`;
        expect(lowerWriteOnlyStorageTextures(wgsl)).toContain("texture_storage_2d<rgba32float, write>");
    });
    it("keeps read_write when the texture is read or the format allows it", () => {
        const read = `${decl("t_0", "rgba16float")}\nfn f() -> vec4<f32> { return textureLoad(t_0, vec2<u32>(0)); }`;
        expect(lowerWriteOnlyStorageTextures(read)).toBe(read);
        const r32 = `${decl("u_0", "r32float")}\nfn g() { textureStore(u_0, vec2<u32>(0), vec4<f32>(1)); }`;
        expect(lowerWriteOnlyStorageTextures(r32)).toBe(r32);
    });
});
