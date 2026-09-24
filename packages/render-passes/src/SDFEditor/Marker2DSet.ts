/**
 * Mirrors RenderPasses/SDFEditor/Marker2DSet: the 2D GUI markers the SDF editor draws,
 * packed as Marker2DDataBlob {type, payload uint[20]} with each marker struct laid out
 * field after field (what the shader's reinterpret<> reads back).
 */

import { Buffer, MemoryType, ResourceBindFlags, RuntimeError, type Device, type ShaderVar } from "@web-falcor/falcor";

/** Mirrors SDF2DShapeType (Marker2DTypes.slang). */
export enum SDF2DShapeType {
    Circle,
    Square,
    Diamond,
    Heart,
    Chevron,
    Ring,
    Tag,
    Cross,
    Asterisk,
    Infinity,
    Pin,
    Arrow,
    RoundedBox,
    Triangle,
    RoundedLine,
    Vector,
    MarkerOpMarker,
    ArrowFromTwoTris,
    CircleSector,
}

/** Mirrors ExcludeBorderFlags. */
export enum ExcludeBorderFlags {
    None = 0,
    Top = 1,
    Right = 2,
    Bottom = 4,
    Left = 8,
}

type Vec2 = readonly [number, number];
type Vec4 = readonly [number, number, number, number];

const kBlobWords = 21;

export class Marker2DSet {
    private markers: Uint32Array[] = [];
    private buffer: Buffer | null = null;
    private dirty = true;

    constructor(
        private readonly device: Device,
        private readonly maxMarkerCount: number,
    ) {}

    get markerCount(): number {
        return this.markers.length;
    }

    /** Mirrors addMarker: `fields` are the payload's words in declaration order. */
    private add(type: SDF2DShapeType, fields: (number | { u: number })[]): void {
        if (this.markers.length >= this.maxMarkerCount) throw new RuntimeError("Number of markers exceeds the maximum number allowed!");
        const blob = new Uint32Array(kBlobWords);
        const f = new Float32Array(blob.buffer);
        blob[0] = type;
        fields.forEach((v, i) => {
            if (typeof v === "number") f[1 + i] = v;
            else blob[1 + i] = v.u;
        });
        this.markers.push(blob);
        this.dirty = true;
    }

    clear(): void {
        this.markers = [];
        this.dirty = true;
    }

    addSimpleMarker(markerType: SDF2DShapeType, size: number, pos: Vec2, rotation: number, color: Vec4): void {
        this.add(markerType, [size, rotation, pos[0], pos[1], ...color]);
    }

    addRoundedLine(posA: Vec2, posB: Vec2, lineWidth: number, color: Vec4): void {
        this.add(SDF2DShapeType.RoundedLine, [...posA, ...posB, lineWidth, ...color]);
    }

    addVector(posA: Vec2, posB: Vec2, lineWidth: number, arrowHeight: number, color: Vec4): void {
        this.add(SDF2DShapeType.Vector, [...posA, ...posB, lineWidth, arrowHeight, ...color]);
    }

    addTriangle(posA: Vec2, posB: Vec2, posC: Vec2, color: Vec4): void {
        this.add(SDF2DShapeType.Triangle, [...posA, ...posB, ...posC, ...color]);
    }

    addRoundedBox(pos: Vec2, halfSides: Vec2, radius: number, rotation: number, color: Vec4): void {
        this.add(SDF2DShapeType.RoundedBox, [radius, rotation, pos[0], pos[1], ...halfSides, ...color]);
    }

    addMarkerOpMarker(op: number, typeA: SDF2DShapeType, posA: Vec2, markerSizeA: number, typeB: SDF2DShapeType, posB: Vec2, markerSizeB: number, color: Vec4, dimmedColor: Vec4): void {
        this.add(SDF2DShapeType.MarkerOpMarker, [{ u: op }, ...posA, { u: typeA }, markerSizeA, ...posB, { u: typeB }, markerSizeB, ...color, ...dimmedColor]);
    }

    addArrowFromTwoTris(startPos: Vec2, endPos: Vec2, headLength: number, headWidth: number, shaftWidth: number, color: Vec4): void {
        this.add(SDF2DShapeType.ArrowFromTwoTris, [...startPos, ...endPos, shaftWidth, headLength, headWidth, ...color]);
    }

    addCircleSector(pos: Vec2, rotation: number, angle: number, minRadius: number, maxRadius: number, color: Vec4, borderColorXYZThicknessW: Vec4, excludeBorderFlags: ExcludeBorderFlags): void {
        this.add(SDF2DShapeType.CircleSector, [...pos, rotation, angle * 0.5, maxRadius, minRadius, ...color, ...borderColorXYZThicknessW, { u: excludeBorderFlags }]);
    }

    /** Mirrors bindShaderData (an empty set binds a one-element placeholder; WebGPU needs a buffer). */
    bindShaderData(v: ShaderVar): void {
        this.updateBuffer();
        v["markers"] = this.buffer!;
        v["markerCount"] = this.markers.length;
    }

    private updateBuffer(): void {
        if (this.buffer && !this.dirty) return;
        this.dirty = false;
        const count = Math.max(1, this.markers.length);
        if (!this.buffer || this.buffer.elementCount < count) {
            this.buffer = new Buffer(this.device, { size: count * kBlobWords * 4, structSize: kBlobWords * 4, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "Marker2DSet::mpMarkerBuffer" });
        }
        if (this.markers.length === 0) return;
        const data = new Uint32Array(this.markers.length * kBlobWords);
        this.markers.forEach((m, i) => data.set(m, i * kBlobWords));
        this.buffer.setBlob(data);
    }
}
