/**
 * Mirrors Falcor/Utils/Math/Rectangle.h (and its python binding): an axis-aligned UV tile,
 * invalid while any minPoint component exceeds maxPoint, starting as (+inf, -inf).
 */

import { float2 } from "./Vector.js";

type Vec2 = { x: number; y: number };
const f = Math.fround;
const v2 = (v: Vec2) => new float2(Number(v.x), Number(v.y));

export class Rectangle {
    minPoint = new float2(Infinity, Infinity);
    maxPoint = new float2(-Infinity, -Infinity);

    constructor(minPoint?: Vec2, maxPoint?: Vec2) {
        if (minPoint) this.set(minPoint, maxPoint ?? minPoint);
    }

    set(pmin: Vec2, pmax: Vec2 = pmin): void {
        this.minPoint = v2(pmin);
        this.maxPoint = v2(pmax);
    }
    invalidate(): void {
        this.minPoint = new float2(Infinity, Infinity);
        this.maxPoint = new float2(-Infinity, -Infinity);
    }
    get valid(): boolean {
        return this.maxPoint.x >= this.minPoint.x && this.maxPoint.y >= this.minPoint.y;
    }
    /** Grows to include a point or another rectangle. */
    include(b: Vec2 | Rectangle): this {
        const [lo, hi] = b instanceof Rectangle ? [b.minPoint, b.maxPoint] : [b, b];
        this.minPoint = new float2(Math.min(this.minPoint.x, lo.x), Math.min(this.minPoint.y, lo.y));
        this.maxPoint = new float2(Math.max(this.maxPoint.x, hi.x), Math.max(this.maxPoint.y, hi.y));
        return this;
    }
    intersection(b: Rectangle): this {
        this.minPoint = new float2(Math.max(this.minPoint.x, b.minPoint.x), Math.max(this.minPoint.y, b.minPoint.y));
        this.maxPoint = new float2(Math.min(this.maxPoint.x, b.maxPoint.x), Math.min(this.maxPoint.y, b.maxPoint.y));
        return this;
    }
    /** True if the intersection has positive area (touching edges don't overlap). */
    overlaps(b: Rectangle): boolean {
        const t = b.clone().intersection(this);
        return t.valid && t.area > 0;
    }
    contains(b: Rectangle): boolean {
        return this.clone().include(b).equals(this);
    }
    get center(): float2 {
        return new float2(f(f(this.minPoint.x + this.maxPoint.x) * 0.5), f(f(this.minPoint.y + this.maxPoint.y) * 0.5));
    }
    get extent(): float2 {
        return new float2(f(this.maxPoint.x - this.minPoint.x), f(this.maxPoint.y - this.minPoint.y));
    }
    get area(): number {
        const e = this.extent;
        return f(e.x * e.y);
    }
    get radius(): number {
        const e = this.extent;
        return f(0.5 * f(Math.sqrt(f(f(e.x * e.x) + f(e.y * e.y)))));
    }
    equals(b: Rectangle): boolean {
        return this.minPoint.x === b.minPoint.x && this.minPoint.y === b.minPoint.y && this.maxPoint.x === b.maxPoint.x && this.maxPoint.y === b.maxPoint.y;
    }
    clone(): Rectangle {
        const r = new Rectangle();
        r.minPoint = new float2(this.minPoint.x, this.minPoint.y);
        r.maxPoint = new float2(this.maxPoint.x, this.maxPoint.y);
        return r;
    }
    /** Python's snake_case fields. */
    get min_point(): float2 { return this.minPoint; }
    set min_point(v: Vec2) { this.minPoint = v2(v); }
    get max_point(): float2 { return this.maxPoint; }
    set max_point(v: Vec2) { this.maxPoint = v2(v); }
}
