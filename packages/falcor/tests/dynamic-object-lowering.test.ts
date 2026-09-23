import { describe, expect, it } from "vitest";
import { lowerDynamicObjects } from "../src/Core/Program/SlangCompiler.js";

describe("lowerDynamicObjects", () => {
    const conformances = [
        { typeName: "B", interfaceName: "I", id: 7 },
        { typeName: "A", interfaceName: "I", id: 3 },
    ];
    it("rewrites createDynamicObject into a switch over the registered conformances", () => {
        const out = lowerDynamicObjects("let o = createDynamicObject<I, int2>(type, v);", conformances);
        expect(out).toContain("let o = __webfalcor_createDynamicObject_I<int2>(type, v);");
        expect(out).toContain("case 3: return reinterpret<A, T>(data);");
        expect(out).toContain("case 7: return reinterpret<B, T>(data);");
        expect(out).toContain("default: return reinterpret<A, T>(data);");
    });
    it("leaves sources alone without conformances or calls", () => {
        expect(lowerDynamicObjects("createDynamicObject<I, int>(0, 0)", [])).toBe("createDynamicObject<I, int>(0, 0)");
        expect(lowerDynamicObjects("void f() {}", conformances)).toBe("void f() {}");
    });
});
