/**
 * Vector math mirroring Falcor/Utils/Math/Vector.h (float2/3/4, column-vector
 * convention). Operations return new instances; hot paths can use the *To
 * variants later if profiling demands it.
 */

export class float2 {
    constructor(
        public x = 0,
        public y = 0,
    ) {}
    static splat(v: number): float2 { return new float2(v, v); }
    clone(): float2 { return new float2(this.x, this.y); }
    toArray(): [number, number] { return [this.x, this.y]; }
}

export class float3 {
    constructor(
        public x = 0,
        public y = 0,
        public z = 0,
    ) {}
    static splat(v: number): float3 { return new float3(v, v, v); }
    clone(): float3 { return new float3(this.x, this.y, this.z); }
    toArray(): [number, number, number] { return [this.x, this.y, this.z]; }
}

export class float4 {
    constructor(
        public x = 0,
        public y = 0,
        public z = 0,
        public w = 0,
    ) {}
    static splat(v: number): float4 { return new float4(v, v, v, v); }
    clone(): float4 { return new float4(this.x, this.y, this.z, this.w); }
    toArray(): [number, number, number, number] { return [this.x, this.y, this.z, this.w]; }
    get xyz(): float3 { return new float3(this.x, this.y, this.z); }
}

// ---- float3 ops (the workhorse) ----

export function add3(a: float3, b: float3): float3 { return new float3(a.x + b.x, a.y + b.y, a.z + b.z); }
export function sub3(a: float3, b: float3): float3 { return new float3(a.x - b.x, a.y - b.y, a.z - b.z); }
export function mul3(a: float3, b: float3 | number): float3 {
    return typeof b === "number" ? new float3(a.x * b, a.y * b, a.z * b) : new float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
export function dot3(a: float3, b: float3): number { return a.x * b.x + a.y * b.y + a.z * b.z; }
export function cross(a: float3, b: float3): float3 {
    return new float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
export function length3(a: float3): number { return Math.hypot(a.x, a.y, a.z); }
export function normalize3(a: float3): float3 {
    const l = length3(a);
    return l > 0 ? mul3(a, 1 / l) : new float3(0, 0, 0);
}
export function lerp3(a: float3, b: float3, t: number): float3 {
    return new float3(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t);
}
export function min3(a: float3, b: float3): float3 { return new float3(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.min(a.z, b.z)); }
export function max3(a: float3, b: float3): float3 { return new float3(Math.max(a.x, b.x), Math.max(a.y, b.y), Math.max(a.z, b.z)); }

// ---- float4 / float2 essentials ----

export function dot4(a: float4, b: float4): number { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
export function mul4(a: float4, s: number): float4 { return new float4(a.x * s, a.y * s, a.z * s, a.w * s); }
export function add2(a: float2, b: float2): float2 { return new float2(a.x + b.x, a.y + b.y); }
export function mul2(a: float2, s: number): float2 { return new float2(a.x * s, a.y * s); }

/** math::frac (component-wise fractional part, matches HLSL frac for the sample patterns). */
export function frac(v: number): number {
    return v - Math.floor(v);
}
