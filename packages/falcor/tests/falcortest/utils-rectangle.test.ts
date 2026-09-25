/**
 * Transplant of FalcorTest Utils/RectangleTests.cpp. Like native, include/intersection mutate
 * the rectangle and return it; native's copies (`Rectangle t = ...`) are clone() here.
 */

import { describe, expect, it } from "vitest";
import { Rectangle } from "../../src/Utils/Math/Rectangle.js";
import { float2 } from "../../src/Utils/Math/Vector.js";

const s = (v: number) => new float2(v, v);
const eq2 = (a: float2, b: float2) => expect([a.x, a.y]).toEqual([b.x, b.y]);

describe("RectangleTests", () => {
    it("Rectangle_Constructors", () => {
        const tile0 = new Rectangle();
        expect(tile0.valid).toBe(false);
        const tile1 = new Rectangle(s(0.5));
        expect(tile1.valid).toBe(true);
        eq2(tile1.center, s(0.5));
        eq2(tile1.extent, s(0));
        tile0.set(s(0.5));
        expect(tile0.equals(tile1)).toBe(true);
        expect(tile0.equals(tile0.intersection(tile1))).toBe(true);
    });

    it("Rectangle_Comparisons", () => {
        const tile0 = new Rectangle();
        expect(tile0.valid).toBe(false);
        tile0.include(s(-1));
        expect(tile0.valid).toBe(true);
        expect(tile0.area).toBe(0);
        tile0.include(s(1));
        expect(tile0.valid).toBe(true);
        expect(tile0.area).toBe(4);
        eq2(tile0.extent, s(2));
        const tile1 = new Rectangle(s(0), s(2));
        const tile2 = tile0.intersection(tile1).clone();
        const tile3 = tile1.intersection(tile2).clone();
        expect(tile2.equals(tile3)).toBe(true);
        eq2(tile2.maxPoint, s(1));
        eq2(tile2.minPoint, s(0));
    });

    it("Rectangle_Contains", () => {
        expect(new Rectangle().valid).toBe(false);
        const big = new Rectangle(s(-1), s(1));
        const small0 = new Rectangle(s(0), s(1));
        const small1 = new Rectangle(s(-1), s(0));
        const small2 = new Rectangle(s(Math.fround(-1.1)), s(0));
        expect(big.contains(big)).toBe(true);
        expect(big.contains(small0)).toBe(true);
        expect(big.contains(small1)).toBe(true);
        expect(big.contains(small2)).toBe(false);
        const invalid0 = new Rectangle();
        for (const r of [small0, small1, small2]) {
            expect(invalid0.overlaps(r)).toBe(false);
            expect(r.overlaps(invalid0)).toBe(false);
        }
        expect(invalid0.overlaps(new Rectangle())).toBe(false);
    });

    it("Rectangle_Overlaps", () => {
        const tile0 = new Rectangle(s(-1), s(1));
        const tile1 = new Rectangle(s(0), s(2));
        const tile2 = new Rectangle(s(1), s(2));
        expect(tile0.overlaps(tile0)).toBe(true);
        expect(tile0.overlaps(tile1)).toBe(true);
        expect(tile1.overlaps(tile0)).toBe(true);
        expect(tile0.overlaps(tile2)).toBe(false);
        expect(tile2.overlaps(tile0)).toBe(false);
        const invalid0 = new Rectangle();
        for (const r of [tile0, tile1, tile2]) {
            expect(invalid0.overlaps(r)).toBe(false);
            expect(r.overlaps(invalid0)).toBe(false);
        }
        expect(invalid0.overlaps(new Rectangle())).toBe(false);
    });
});
