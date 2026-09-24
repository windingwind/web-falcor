/**
 * Axis-aligned bounding box mirroring Falcor/Utils/Math/AABB.h (and its python binding): invalid
 * while any minPoint component exceeds maxPoint, starting as (+inf, -inf).
 */

import { float3 } from "./Vector.js";

type Vec = { x: number; y: number; z: number };
const v3 = (v: Vec) => new float3(Number(v.x), Number(v.y), Number(v.z));

export class AABB {
    minPoint = new float3(Infinity, Infinity, Infinity);
    maxPoint = new float3(-Infinity, -Infinity, -Infinity);

    constructor(minPoint?: Vec, maxPoint?: Vec) {
        if (minPoint) this.minPoint = v3(minPoint);
        if (maxPoint) this.maxPoint = v3(maxPoint ?? minPoint!);
        else if (minPoint) this.maxPoint = v3(minPoint);
    }

    /** Aliases for the builder's custom-primitive records. */
    get min(): float3 { return this.minPoint; }
    get max(): float3 { return this.maxPoint; }

    get valid(): boolean {
        return this.maxPoint.x >= this.minPoint.x && this.maxPoint.y >= this.minPoint.y && this.maxPoint.z >= this.minPoint.z;
    }
    get center(): float3 {
        return new float3((this.minPoint.x + this.maxPoint.x) * 0.5, (this.minPoint.y + this.maxPoint.y) * 0.5, (this.minPoint.z + this.maxPoint.z) * 0.5);
    }
    get extent(): float3 {
        return new float3(this.maxPoint.x - this.minPoint.x, this.maxPoint.y - this.minPoint.y, this.maxPoint.z - this.minPoint.z);
    }
    get area(): number {
        const e = this.extent;
        return (e.x * e.y + e.x * e.z + e.y * e.z) * 2;
    }
    get volume(): number {
        const e = this.extent;
        return e.x * e.y * e.z;
    }
    /** Radius of the smallest enclosing sphere. */
    get radius(): number {
        const e = this.extent;
        return 0.5 * Math.hypot(e.x, e.y, e.z);
    }

    invalidate(): void {
        this.minPoint = new float3(Infinity, Infinity, Infinity);
        this.maxPoint = new float3(-Infinity, -Infinity, -Infinity);
    }
    /** Grows the box to a point or another box. */
    include(p: Vec | AABB): AABB {
        const [lo, hi] = p instanceof AABB ? [p.minPoint, p.maxPoint] : [p, p];
        this.minPoint = new float3(Math.min(this.minPoint.x, lo.x), Math.min(this.minPoint.y, lo.y), Math.min(this.minPoint.z, lo.z));
        this.maxPoint = new float3(Math.max(this.maxPoint.x, hi.x), Math.max(this.maxPoint.y, hi.y), Math.max(this.maxPoint.z, hi.z));
        return this;
    }
    /** Shrinks the box to its intersection with b (invalid when they don't overlap). */
    intersection(b: AABB): AABB {
        this.minPoint = new float3(Math.max(this.minPoint.x, b.minPoint.x), Math.max(this.minPoint.y, b.minPoint.y), Math.max(this.minPoint.z, b.minPoint.z));
        this.maxPoint = new float3(Math.min(this.maxPoint.x, b.maxPoint.x), Math.min(this.maxPoint.y, b.maxPoint.y), Math.min(this.maxPoint.z, b.maxPoint.z));
        return this;
    }
}
