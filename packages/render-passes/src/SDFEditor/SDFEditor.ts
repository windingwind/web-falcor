/**
 * Mirrors Source/RenderPasses/SDFEditor: interactive editing of SBS SDF grids. A full-screen
 * GUI pass (GUIPass.ps.slang, unmodified) composites the grid bounding boxes, grid and
 * symmetry planes, the primitive preview and the 2D GUI markers (current-mode badge,
 * selection wheel) over the input color, and records picking info at the mouse.
 *
 * Input mirrors onKeyEvent/onMouseEvent: Tab opens the shape/operation wheel, Alt previews
 * and places primitives (Alt+LMB adds), Ctrl+R/S rotate/scale the primitive, Shift+T/R/S
 * transform the instance, G/H/B/X/C toggle the grid plane, symmetry, bounding boxes,
 * preview and editing on other surfaces, and Ctrl+Z/Y undo/redo.
 *
 * §9: picking info is read back asynchronously (native waits on a fence each frame), so
 * picks use the previous completed readback; file dialogs download instead of saving.
 */

import {
    Buffer,
    Fbo,
    FullScreenPass,
    Logger,
    MemoryType,
    Properties,
    RenderData,
    RenderPass,
    RenderPassReflection,
    ResourceBindFlags,
    ResourceFormat,
    ResourceType,
    SDF3DShapeType,
    SDFGridPrimitives,
    SDFOperationType,
    Texture,
    float3,
    float4x4,
    inverse,
    matrixFromQuat,
    mulQuat,
    quatFromAngleAxis,
    quatf,
    registerRenderPass,
    transformPoint,
    transformVector,
    transpose,
    type CompileData,
    type Device,
    type RenderContext,
    type SDF3DPrimitive,
    type ShaderVar,
    type UIWidgets,
} from "@web-falcor/falcor";
import { Marker2DSet, SDF2DShapeType } from "./Marker2DSet.js";
import { SelectionWheel, kInvalidIndex, type SelectionWheelDesc } from "./SelectionWheel.js";

const kGUIPassShaderFilename = "RenderPasses/SDFEditor/GUIPass.ps.slang";
const kInvalidPrimitiveID = 0xffffffff;

type V2 = [number, number];
type V3 = [number, number, number];
type V4 = [number, number, number, number];

const kLineColor: V4 = [0.585, 1.0, 0.0, 1.0];
const kMarkerColor: V4 = [0.9, 0.9, 0.9, 1.0];
const kSelectionColor: V4 = [1.0, 1.0, 1.0, 0.75];
const kCurrentModeBGColor: V4 = [0.585, 1.0, 0.0, 0.5];
const kFadeAwayDuration = 0.25;
const kMarkerSizeFactor = 0.75;
const kMarkerSizeFactorSmoothUnion = 0.6;
const kScrollTranslationMultiplier = 0.01;
const kMaxOpSmoothingRadius = 0.01;
const kMinOpSmoothingRadius = 0.0001;
const kMinShapeBlobbyness = 0.001;
const kMaxShapeBlobbyness = 0.02;
const kMinOperationSmoothness = 0.01;
const kMaxOperationSmoothness = 0.05;

/** Mirrors SDFBBRenderMode. */
export enum SDFBBRenderMode {
    Disabled = 0,
    RenderAll = 1,
    RenderSelectedOnly = 2,
    Count = 3,
}

/** Mirrors SDFEditorAxis. */
enum SDFEditorAxis {
    X = 0,
    Y = 1,
    Z = 2,
    OpSmoothing = 3,
    All = 4,
    Count = 5,
}

enum TransformationState {
    None,
    Translating,
    Rotating,
    Scaling,
}

/** Mirrors SDFGridPlane (defaults as native). */
interface SDFGridPlane {
    position: V3;
    gridLineWidth: number;
    normal: V3;
    gridScale: number;
    rightVector: V3;
    planeSize: number;
    color: V4;
    active: boolean;
}

const newPlane = (): SDFGridPlane => ({ position: [0, 0, 0], gridLineWidth: 0.04, normal: [0, 0, 1], gridScale: 25, rightVector: [1, 0, 0], planeSize: 0.25, color: [0, 0.5, 1, 0.5], active: false });
const copyPlane = (p: SDFGridPlane): SDFGridPlane => ({ ...p, position: [...p.position], normal: [...p.normal], rightVector: [...p.rightVector], color: [...p.color] });

// --- small vector / matrix helpers ([x,y,z] tuples; row-major 3x3 as 9 numbers) ---
const add3 = (a: V3, b: V3): V3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub3 = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const scale3 = (a: V3, s: number): V3 => [a[0] * s, a[1] * s, a[2] * s];
const dot3 = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross3 = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const len3 = (a: V3) => Math.hypot(a[0], a[1], a[2]);
const norm3 = (a: V3): V3 => scale3(a, 1 / len3(a));
const f3 = (v: V3) => new float3(v[0], v[1], v[2]);
const t3 = (v: float3): V3 => [v.x, v.y, v.z];
const reflect3 = (i: V3, n: V3): V3 => sub3(i, scale3(n, 2 * dot3(n, i)));

/** The upper-left 3x3 of a float4x4, row-major. */
const mat3 = (m: float4x4): number[] => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => m.get(r, c)));
const mat4 = (m: number[]): float4x4 => {
    const out = float4x4.identity();
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) out.set(r, c, m[r * 3 + c]!);
    return out;
};
const inverse3 = (m: number[]) => mat3(inverse(mat4(m)));
const transpose3 = (m: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => m[c * 3 + r]!));
const mul3 = (a: number[], b: number[]) => [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => a[r * 3]! * b[c]! + a[r * 3 + 1]! * b[3 + c]! + a[r * 3 + 2]! * b[6 + c]!));

/** Transform (T * R * S) with decomposition, like native's Transform/math::decompose. */
interface TRS {
    t: V3;
    r: quatf;
    s: V3;
}
function quatFromRotation(m: number[]): quatf {
    const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m as [number, number, number, number, number, number, number, number, number];
    const trace = m00 + m11 + m22;
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1);
        return new quatf((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s);
    }
    if (m00 > m11 && m00 > m22) {
        const s = 2 * Math.sqrt(1 + m00 - m11 - m22);
        return new quatf(0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s);
    }
    if (m11 > m22) {
        const s = 2 * Math.sqrt(1 + m11 - m00 - m22);
        return new quatf((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s);
    }
    const s = 2 * Math.sqrt(1 + m22 - m00 - m11);
    return new quatf((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s);
}
function decompose(m: float4x4): TRS {
    const col = (c: number): V3 => [m.get(0, c), m.get(1, c), m.get(2, c)];
    const s: V3 = [len3(col(0)), len3(col(1)), len3(col(2))];
    const rot = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => m.get(r, c) / s[c]!));
    return { t: col(3), r: quatFromRotation(rot), s };
}
function trsMatrix(x: TRS): float4x4 {
    const r = matrixFromQuat(x.r);
    const out = float4x4.identity();
    for (let row = 0; row < 3; row++) {
        for (let c = 0; c < 3; c++) out.set(row, c, r.get(row, c) * x.s[c]!);
        out.set(row, 3, x.t[row]!);
    }
    return out;
}
const rotateAround = (angle: number, axis: V3) => matrixFromQuat(quatFromAngleAxis(angle, f3(norm3(axis))));

const isOperationSmooth = (op: SDFOperationType) => op === SDFOperationType.SmoothUnion || op === SDFOperationType.SmoothSubtraction || op === SDFOperationType.SmoothIntersection;
const sdf3DTo2DShape = (t: SDF3DShapeType) => (t === SDF3DShapeType.Box ? SDF2DShapeType.Square : SDF2DShapeType.Circle);

/** Mirrors SDF3DPrimitiveFactory::computeAABB (grid-local space). */
function computePrimitiveAABB(p: SDF3DPrimitive): { min: V3; max: V3 } {
    const rounding = p.shapeBlobbing + (p.operationType >= SDFOperationType.SmoothUnion ? p.operationSmoothing : 0);
    let corners: V3[];
    const box = (x: number, yLo: number, yHi: number, z: number): V3[] => [-1, 1].flatMap((sx) => [yLo, yHi].flatMap((y) => [-1, 1].map((sz) => [sx * x, y, sz * z] as V3)));
    switch (p.shapeType) {
        case SDF3DShapeType.Sphere: {
            const r = p.shapeData[0] + rounding;
            corners = box(r, -r, r, r);
            break;
        }
        case SDF3DShapeType.Ellipsoid:
        case SDF3DShapeType.Box: {
            const h = p.shapeData.map((v) => v + rounding) as V3;
            corners = box(h[0], -h[1], h[1], h[2]);
            break;
        }
        case SDF3DShapeType.Torus: {
            const big = p.shapeData[0] + rounding;
            corners = box(big, -rounding, rounding, big);
            break;
        }
        case SDF3DShapeType.Cone: {
            const radius = p.shapeData[0] * p.shapeData[1] + rounding;
            corners = box(radius, -rounding, p.shapeData[1] + rounding, radius);
            break;
        }
        case SDF3DShapeType.Capsule: {
            const halfLen = p.shapeData[0] + rounding;
            corners = box(rounding, -halfLen, halfLen, rounding);
            break;
        }
        default:
            throw new Error("SDF Primitive has unknown primitive type");
    }
    // translate * inverse(transpose(invRotationScale)).
    const rs = mat4(inverse3(transpose3(p.invRotationScale)));
    const min: V3 = [Infinity, Infinity, Infinity];
    const max: V3 = [-Infinity, -Infinity, -Infinity];
    for (const c of corners) {
        const w = add3(t3(transformVector(rs, f3(c))), p.translation);
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k]!, w[k]!);
            max[k] = Math.max(max[k]!, w[k]!);
        }
    }
    return { min, max };
}

/** Mirrors SDF3DPrimitiveFactory::initCommon with an identity transform. */
function initPrimitive(shapeType: SDF3DShapeType, shapeData: V3, blobbing: number, smoothing: number, op: SDFOperationType): SDF3DPrimitive {
    return { shapeType, shapeData: [...shapeData], shapeBlobbing: blobbing, operationType: op, operationSmoothing: smoothing, translation: [0, 0, 0], invRotationScale: [1, 0, 0, 0, 1, 0, 0, 0, 1] };
}
const clonePrimitive = (p: SDF3DPrimitive): SDF3DPrimitive => ({ ...p, shapeData: [...p.shapeData], translation: [...p.translation], invRotationScale: [...p.invRotationScale] });

/** Pass-level input events (the viewer forwards canvas mouse/wheel and keyboard). */
export interface SDFEditorMouseEvent {
    type: "buttonDown" | "buttonUp" | "move" | "wheel";
    button?: "left" | "right" | "middle";
    /** Normalized [0, 1], (0, 0) top-left. */
    pos: [number, number];
    wheelDelta?: [number, number];
}
export interface SDFEditorKeyEvent {
    type: "keyPressed" | "keyReleased" | "keyRepeated";
    /** Native Input::Key name ("Tab", "LeftAlt", "G", "Key1", ...). */
    key: string;
    mods?: { shift?: boolean; ctrl?: boolean; alt?: boolean };
}

/** SDFEditingData as a WGSL storage struct: 4 u32, two SDF3DPrimitive (112 B), two AABB (32 B). */
const kEditingDataSize = 304;
function packPrimitive(view: DataView, at: number, p: SDF3DPrimitive): void {
    view.setUint32(at, p.shapeType, true);
    p.shapeData.forEach((v, i) => view.setFloat32(at + 16 + 4 * i, v, true));
    view.setFloat32(at + 28, p.shapeBlobbing, true);
    view.setUint32(at + 32, p.operationType, true);
    view.setFloat32(at + 36, p.operationSmoothing, true);
    p.translation.forEach((v, i) => view.setFloat32(at + 48 + 4 * i, v, true));
    // float3x3: three 16-byte rows.
    for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) view.setFloat32(at + 64 + r * 16 + c * 4, p.invRotationScale[r * 3 + c]!, true);
}
function packAABB(view: DataView, at: number, bb: { min: V3; max: V3 }): void {
    bb.min.forEach((v, i) => view.setFloat32(at + 4 * i, v, true));
    bb.max.forEach((v, i) => view.setFloat32(at + 16 + 4 * i, v, true));
}

/** The SBS grid interface the editor edits (SDFSBS: primitives plus its stored field). */
interface EditableGrid {
    gridWidth: number;
    primitives?: SDFGridPrimitives;
    cornerValues?: () => Float32Array;
    setValues(values: Float32Array, gridWidth: number): void;
}

export class SDFEditor extends RenderPass {
    private guiPass: FullScreenPass | null = null;
    private fbo = new Fbo();
    private editingVBuffer: Texture | null = null;
    private editingLinearZ: Texture | null = null;
    private editingDataBuffer: Buffer;
    private pickingInfo: Buffer;
    private gridInstanceIDsBuffer: Buffer | null = null;
    private gridInstanceCount = 0;
    private pickingReadbackPending = false;
    private picking = { distance: Infinity, instanceID: 0xffffffff, hitType: 0 };
    private frameDim: V2 = [0, 0];

    private readonly markers: Marker2DSet;
    private readonly wheel: SelectionWheel;
    private ui = {
        recordStartingMousePos: false,
        scrollDelta: 0,
        keys: { undo: false, redo: false, shift: false, control: false, prevShift: false, prevControl: false },
        startMousePosition: [0, 0] as V2,
        currentMousePosition: [0, 0] as V2,
        prevMousePosition: [0, 0] as V2,
        timeOfReleaseMainGUIKey: 0,
        fadeAwayGUI: false,
        drawCurrentModes: true,
        currentBlobbing: 0,
        currentEditingShape: SDF3DShapeType.Sphere,
        currentEditingOperator: SDFOperationType.Union,
        bbRenderSettings: { renderMode: SDFBBRenderMode.RenderSelectedOnly as number, selectedInstanceID: 0, edgeThickness: 0.0001 },
        previousGridPlane: newPlane(),
        gridPlane: newPlane(),
        previousSymmetryPlane: newPlane(),
        symmetryPlane: { ...newPlane(), normal: [1, 0, 0] as V3, rightVector: [0, 0, -1] as V3, color: [1, 0.75, 0.8, 0.5] as V4 },
    };
    private currentEdit = {
        instanceID: 0,
        gridID: 0,
        grid: null as EditableGrid | null,
        primitive: initPrimitive(SDF3DShapeType.Sphere, [0.01, 0.01, 0.01], 0, 0, SDFOperationType.Union),
        symmetryPrimitive: initPrimitive(SDF3DShapeType.Sphere, [0.01, 0.01, 0.01], 0, 0, SDFOperationType.Union),
        primitiveID: kInvalidPrimitiveID,
        symmetryPrimitiveID: kInvalidPrimitiveID,
    };
    private primitiveEdit = {
        prevState: TransformationState.None,
        state: TransformationState.None,
        startInstanceTransform: null as TRS | null,
        startPrimitiveTransform: null as TRS | null,
        startPrimitive: null as SDF3DPrimitive | null,
        startPlanePos: [0, 0, 0] as V3,
        referencePlaneDir: [0, 0, 0] as V3,
        axis: SDFEditorAxis.All,
        prevAxis: SDFEditorAxis.All,
        startMousePos: [0, 0] as V2,
    };
    private instanceEdit = {
        prevState: TransformationState.None,
        state: TransformationState.None,
        startTransform: null as TRS | null,
        startPlanePos: [0, 0, 0] as V3,
        referencePlaneDir: [0, 0, 0] as V3,
        prevScrollTotal: 0,
        scrollTotal: 0,
        startMousePos: [0, 0] as V2,
    };
    private performedEdits: { gridID: number; primitiveID: number }[] = [];
    private undoneEdits: { gridID: number; primitive: SDF3DPrimitive }[] = [];
    private lmbDown = false;
    private rmbDown = false;
    private mmbDown = false;
    private editingKeyDown = false;
    private guiKeyDown = false;
    private previewEnabled = true;
    private allowEditingOnOtherSurfaces = false;
    private autoBakingEnabled = true;
    private nonBakedPrimitiveCount = 0;
    private bakePrimitivesBatchSize = 5;
    private preservedHistoryCount = 100;
    private undoPressedCount = 0;
    private redoPressedCount = 0;

    constructor(device: Device, _props: Properties) {
        super(device);
        this.markers = new Marker2DSet(device, 100);
        this.wheel = new SelectionWheel(this.markers);
        this.editingDataBuffer = new Buffer(device, { size: kEditingDataSize, structSize: kEditingDataSize, bindFlags: ResourceBindFlags.ShaderResource, memoryType: MemoryType.DeviceLocal, name: "SDFEditor::editingData" });
        this.pickingInfo = new Buffer(device, { size: 12, structSize: 12, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "SDFEditor::pickingInfo" });
    }

    override getProperties(): Properties {
        return new Properties({});
    }

    override reflect(compileData: CompileData): RenderPassReflection {
        const r = new RenderPassReflection();
        const [w, h] = compileData.defaultTexDims;
        r.addInput("vbuffer", "Visibility buffer in packed format").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("linearZ", "Linear Z and slope").bindFlags(ResourceBindFlags.ShaderResource);
        r.addInput("inputColor", "The input image (2D GUI will be drawn on top)").bindFlags(ResourceBindFlags.ShaderResource);
        r.addOutput("output", "Input image with 2D GUI drawn on top").texture2D(w, h).format(ResourceFormat.RGBA32Float).bindFlags(ResourceBindFlags.RenderTarget | ResourceBindFlags.ShaderResource);
        return r;
    }

    override setScene(scene: typeof this.scene): void {
        super.setScene(scene);
        this.guiPass = null;
        this.gridInstanceIDsBuffer = null;
        if (!scene) return;
        const ids = scene.getSDFGridInstanceIDs();
        if (ids.length === 0) return;
        this.ui.currentEditingShape = SDF3DShapeType.Sphere;
        this.ui.currentEditingOperator = SDFOperationType.Union;
        this.currentEdit.instanceID = ids[0]!;
        this.currentEdit.gridID = scene.findSDFGridIDFromGeometryInstanceID(ids[0]!);
        this.currentEdit.grid = scene.sdfGrids[this.currentEdit.gridID]!.grid as unknown as EditableGrid;
        this.currentEdit.primitive = initPrimitive(SDF3DShapeType.Sphere, [0.01, 0.01, 0.01], 0, 0, SDFOperationType.Union);
        const transform = scene.getSDFGridTransform(this.currentEdit.gridID);
        this.ui.bbRenderSettings.selectedInstanceID = this.currentEdit.instanceID;
        this.ui.gridPlane.position = [transform.get(0, 3), transform.get(1, 3), transform.get(2, 3)];
        this.ui.previousGridPlane.position = [...this.ui.gridPlane.position];
        this.updateSymmetryPrimitive();
        this.nonBakedPrimitiveCount = this.gridPrimitives()?.primitiveCount ?? 0;
    }

    /** The edited grid's primitive list (created over its stored values on first edit, like native SBS). */
    private gridPrimitives(create = false): SDFGridPrimitives | undefined {
        const grid = this.currentEdit.grid;
        if (!grid) return undefined;
        if (!grid.primitives && create) grid.primitives = new SDFGridPrimitives(grid, grid.gridWidth, [], grid.cornerValues?.());
        return grid.primitives;
    }

    private gridTransform(): float4x4 {
        return this.scene!.getSDFGridTransform(this.currentEdit.gridID);
    }

    private isMainGUIKeyDown(): boolean {
        return this.guiKeyDown && this.primitiveEdit.state === TransformationState.None && !this.gridPlaneManipulated() && !this.symmetryPlaneManipulated() && !this.ui.keys.undo && !this.ui.keys.redo;
    }

    private gridPlaneManipulated(): boolean {
        return this.ui.gridPlane.active && this.rmbDown;
    }

    private symmetryPlaneManipulated(): boolean {
        return this.ui.symmetryPlane.active && this.mmbDown;
    }

    private updateEditShapeType(): void {
        this.currentEdit.primitive.shapeType = this.ui.currentEditingShape;
        this.currentEdit.primitive.shapeBlobbing = this.ui.currentBlobbing;
        if (this.ui.symmetryPlane.active) this.updateSymmetryPrimitive();
    }

    private updateEditOperationType(): void {
        this.currentEdit.primitive.operationType = this.ui.currentEditingOperator;
        this.currentEdit.primitive.operationSmoothing = isOperationSmooth(this.ui.currentEditingOperator) ? 0.5 * (kMaxOperationSmoothness + kMinOperationSmoothness) : 0;
        if (this.ui.symmetryPlane.active) this.updateSymmetryPrimitive();
    }

    private updateSymmetryPrimitive(): void {
        if (!this.scene) return;
        const invInstance = inverse(this.gridTransform());
        const primPos = this.currentEdit.primitive.translation as V3;
        const planePos = t3(transformPoint(invInstance, f3(this.ui.symmetryPlane.position)));
        const planeNormal = norm3(t3(transformVector(invInstance, f3(this.ui.symmetryPlane.normal))));
        const projPrim = sub3(primPos, scale3(planeNormal, dot3(primPos, planeNormal)));
        const projPlane = scale3(planeNormal, dot3(planePos, planeNormal));
        let reflected = len3(projPrim) > 0 ? reflect3(scale3(sub3(primPos, projPlane), -1), norm3(projPrim)) : scale3(planeNormal, -len3(sub3(primPos, projPlane)));
        reflected = add3(reflected, projPlane);
        // math::rotate(float4x4(inverse(invRotationScale)), pi, normal): the transform times a rotation.
        const symmetric = mul3(inverse3(this.currentEdit.primitive.invRotationScale), mat3(rotateAround(Math.PI, planeNormal)));
        this.currentEdit.symmetryPrimitive = clonePrimitive(this.currentEdit.primitive);
        this.currentEdit.symmetryPrimitive.translation = reflected;
        this.currentEdit.symmetryPrimitive.invRotationScale = inverse3(symmetric);
    }

    private setupPrimitiveAndOperation(center: V2, markerSize: number, shape: SDF3DShapeType, op: SDFOperationType, color: V4, alpha = 1): void {
        const dimmed: V4 = [color[0] * 0.6, color[1] * 0.6, color[2] * 0.6, color[3]];
        let f = markerSize * 0.25;
        f += op === SDFOperationType.SmoothUnion ? f * kMarkerSizeFactor : 0;
        const fade = (c: V4): V4 => [c[0], c[1], c[2], c[3] * alpha];
        this.markers.addMarkerOpMarker(op, sdf3DTo2DShape(shape), [center[0] - f, center[1] - f], markerSize, sdf3DTo2DShape(shape), [center[0] + f, center[1] + f], markerSize, fade(color), fade(dimmed));
    }

    private setupCurrentModes2D(): void {
        const side = Math.min(this.frameDim[0], this.frameDim[1]) / 6;
        const markerSize = (side * 0.5) / kMarkerSizeFactor;
        const cornerOffset = 10;
        const center: V2 = [side * 0.5 + cornerOffset, this.frameDim[1] - side * 0.5 - cornerOffset];
        this.markers.addRoundedBox(center, [side * 0.5, side * 0.5], markerSize * 0.15, 0, kCurrentModeBGColor);
        const f = this.ui.currentEditingOperator === SDFOperationType.SmoothUnion ? kMarkerSizeFactorSmoothUnion : kMarkerSizeFactor;
        this.setupPrimitiveAndOperation(center, markerSize * f, this.ui.currentEditingShape, this.ui.currentEditingOperator, kMarkerColor);
    }

    private setup2DGUI(): void {
        const center: V2 = [this.frameDim[0] * 0.5, this.frameDim[1] * 0.5];
        const radius = 0.5 * Math.min(this.frameDim[0], this.frameDim[1]);
        const markerSize = radius * 0.2;
        this.markers.clear();

        if (this.isMainGUIKeyDown() || this.ui.fadeAwayGUI) {
            let alpha = 1;
            if (!this.isMainGUIKeyDown()) {
                const dt = (performance.now() - this.ui.timeOfReleaseMainGUIKey) / 1000;
                if (dt >= kFadeAwayDuration) {
                    this.ui.fadeAwayGUI = false;
                    alpha = 0;
                } else alpha = 1 - dt / kFadeAwayDuration;
            }
            const color: V4 = [kMarkerColor[0], kMarkerColor[1], kMarkerColor[2], kMarkerColor[3] * alpha];
            const desc: SelectionWheelDesc = {
                position: center,
                minRadius: 0.3 * radius,
                maxRadius: 0.7 * radius,
                baseColor: [0.13, 0.13, 0.1523, 0.8 * alpha],
                highlightColor: [0.4648, 0.7226, 0.0, 0.8 * alpha],
                sectorGroups: [2, 4],
                lineColor: [kLineColor[0], kLineColor[1], kLineColor[2], kLineColor[3] * alpha],
                borderWidth: 10,
            };
            this.wheel.update(this.ui.currentMousePosition, desc);
            this.markers.addSimpleMarker(SDF2DShapeType.Square, markerSize, this.wheel.getCenterPositionOfSector(0, 0), 0, color);
            this.markers.addSimpleMarker(SDF2DShapeType.Circle, markerSize * 0.5, this.wheel.getCenterPositionOfSector(0, 1), 0, color);
            const ops = [SDFOperationType.SmoothSubtraction, SDFOperationType.Subtraction, SDFOperationType.Union, SDFOperationType.SmoothUnion];
            ops.forEach((op, i) => this.setupPrimitiveAndOperation(this.wheel.getCenterPositionOfSector(1, i), markerSize * (op === SDFOperationType.SmoothUnion ? kMarkerSizeFactorSmoothUnion : kMarkerSizeFactor), this.ui.currentEditingShape, op, kMarkerColor, alpha));
        }

        const kShapeTypes = [SDF3DShapeType.Box, SDF3DShapeType.Sphere];
        const kOperationTypes = [SDFOperationType.SmoothSubtraction, SDFOperationType.Subtraction, SDFOperationType.Union, SDFOperationType.SmoothUnion];
        const moved = this.ui.startMousePosition[0] !== this.ui.currentMousePosition[0] || this.ui.startMousePosition[1] !== this.ui.currentMousePosition[1];
        if (this.isMainGUIKeyDown() && !this.ui.recordStartingMousePos && moved) {
            const shapes = this.wheel.isMouseOnGroup(this.ui.currentMousePosition, 0);
            if (shapes.inGroup && shapes.sectorIndex !== kInvalidIndex && shapes.sectorIndex < kShapeTypes.length) {
                const type = kShapeTypes[shapes.sectorIndex]!;
                if (type === SDF3DShapeType.Box) this.markers.addSimpleMarker(SDF2DShapeType.Square, markerSize, center, 0, kSelectionColor);
                else this.markers.addSimpleMarker(SDF2DShapeType.Circle, markerSize * 0.5, center, 0, kSelectionColor);
                if (this.lmbDown) {
                    this.ui.currentEditingShape = type;
                    this.updateEditShapeType();
                }
            }
            const opsHit = this.wheel.isMouseOnGroup(this.ui.currentMousePosition, 1);
            if (opsHit.inGroup && opsHit.sectorIndex !== kInvalidIndex && opsHit.sectorIndex < kOperationTypes.length) {
                const op = kOperationTypes[opsHit.sectorIndex]!;
                this.setupPrimitiveAndOperation(center, markerSize * (op === SDFOperationType.SmoothUnion ? kMarkerSizeFactorSmoothUnion : kMarkerSizeFactor), this.ui.currentEditingShape, op, kSelectionColor);
                if (this.lmbDown) {
                    this.ui.currentEditingOperator = op;
                    this.updateEditOperationType();
                }
            }
        }
        if (this.ui.drawCurrentModes) this.setupCurrentModes2D();
    }

    /** Mirrors Camera::computeRayPinhole (no jitter) at the mouse pixel. */
    private mouseRay(): { origin: V3; dir: V3 } {
        const cam = this.scene!.camera.getData();
        const px = Math.trunc(this.ui.currentMousePosition[0]);
        const py = Math.trunc(this.ui.currentMousePosition[1]);
        const nx = ((px + 0.5) / this.frameDim[0]) * 2 - 1;
        const ny = 1 - ((py + 0.5) / this.frameDim[1]) * 2;
        return { origin: t3(cam.posW), dir: norm3(add3(add3(scale3(t3(cam.cameraU), nx), scale3(t3(cam.cameraV), ny)), t3(cam.cameraW))) };
    }

    private cameraPlaneNormal(): V3 {
        const c = this.scene!.camera;
        return scale3(norm3(sub3(t3(c.getTarget()), t3(c.getPosition()))), -1);
    }

    private referenceDir(planeNormal: V3): V3 {
        const arbitrary: V3 = Math.abs(planeNormal[2]) < 1.1920929e-7 ? [0, 0, 1] : [1, 0, 0];
        return norm3(sub3(arbitrary, scale3(planeNormal, dot3(arbitrary, planeNormal))));
    }

    private planeAngle(planeNormal: V3, dir: V3, reference: V3): number {
        return Math.atan2(dot3(planeNormal, cross3(dir, reference)), dot3(dir, reference));
    }

    private handleActions(): void {
        if (!this.scene || !this.currentEdit.grid) return;
        const instanceTransform = this.gridTransform();
        const ray = this.mouseRay();
        const mouseMoved = this.ui.currentMousePosition[0] !== this.ui.prevMousePosition[0] || this.ui.currentMousePosition[1] !== this.ui.prevMousePosition[1];

        if (this.instanceEdit.state !== TransformationState.None) {
            if (this.instanceEdit.prevState === TransformationState.None) {
                this.instanceEdit.scrollTotal = 0;
                this.instanceEdit.startTransform = decompose(instanceTransform);
                const planeNormal = this.cameraPlaneNormal();
                const startT = dot3(sub3(this.instanceEdit.startTransform.t, ray.origin), planeNormal) / dot3(ray.dir, planeNormal);
                this.instanceEdit.startPlanePos = add3(ray.origin, scale3(ray.dir, startT));
                this.instanceEdit.startMousePos = [...this.ui.currentMousePosition];
                this.instanceEdit.referencePlaneDir = this.referenceDir(planeNormal);
            }
            if (mouseMoved || this.instanceEdit.prevScrollTotal !== this.instanceEdit.scrollTotal) {
                const start = this.instanceEdit.startTransform!;
                const planeNormal = this.cameraPlaneNormal();
                const t = dot3(sub3(start.t, ray.origin), planeNormal) / dot3(ray.dir, planeNormal);
                const final: TRS = { t: [...start.t], r: start.r, s: [...start.s] };
                if (this.instanceEdit.state === TransformationState.Translating) {
                    final.t = add3(ray.origin, scale3(ray.dir, t + this.instanceEdit.scrollTotal * kScrollTranslationMultiplier));
                } else if (this.instanceEdit.state === TransformationState.Rotating) {
                    const p = add3(ray.origin, scale3(ray.dir, t));
                    const startDir = norm3(sub3(this.instanceEdit.startPlanePos, start.t));
                    const currDir = norm3(sub3(p, start.t));
                    const deltaAngle = this.planeAngle(planeNormal, startDir, this.instanceEdit.referencePlaneDir) - this.planeAngle(planeNormal, currDir, this.instanceEdit.referencePlaneDir);
                    const localNormal = norm3(t3(transformVector(inverse(trsMatrix(start)), f3(planeNormal))));
                    final.r = mulQuat(start.r, quatFromAngleAxis(deltaAngle, f3(localNormal)));
                } else if (this.instanceEdit.state === TransformationState.Scaling) {
                    const deltaX = this.ui.currentMousePosition[0] - this.instanceEdit.startMousePos[0];
                    const scale = Math.pow(deltaX < 0 ? 1 / 1.05 : 1.05, (100 * Math.abs(deltaX)) / this.frameDim[0]);
                    final.s = scale3(start.s, scale);
                }
                this.scene.updateSDFGridTransform(this.currentEdit.gridID, trsMatrix(final));
            }
        } else if (this.primitiveEdit.state !== TransformationState.None) {
            const resetStart = () => {
                const planeOrigin = t3(transformPoint(trsMatrix(this.primitiveEdit.startInstanceTransform!), f3(this.currentEdit.primitive.translation as V3)));
                const planeNormal = this.cameraPlaneNormal();
                const startT = dot3(sub3(planeOrigin, ray.origin), planeNormal) / dot3(ray.dir, planeNormal);
                this.primitiveEdit.startPlanePos = add3(ray.origin, scale3(ray.dir, startT));
                this.primitiveEdit.startMousePos = [...this.ui.currentMousePosition];
                this.primitiveEdit.referencePlaneDir = this.referenceDir(planeNormal);
            };
            if (this.primitiveEdit.prevState === TransformationState.None) {
                this.primitiveEdit.startPrimitive = clonePrimitive(this.currentEdit.primitive);
                this.primitiveEdit.startInstanceTransform = decompose(instanceTransform);
                const primTransform = decompose(mat4(inverse3(transpose3(this.currentEdit.primitive.invRotationScale))));
                this.primitiveEdit.startPrimitiveTransform = { t: [...(this.currentEdit.primitive.translation as V3)], r: primTransform.r, s: primTransform.s };
                resetStart();
            } else if (this.primitiveEdit.state !== this.primitiveEdit.prevState || this.primitiveEdit.axis !== this.primitiveEdit.prevAxis) {
                this.currentEdit.primitive = clonePrimitive(this.primitiveEdit.startPrimitive!);
                if (this.ui.symmetryPlane.active) this.updateSymmetryPrimitive();
                resetStart();
            }
            if (mouseMoved) {
                const startPrim = this.primitiveEdit.startPrimitiveTransform!;
                const planeOrigin = t3(transformPoint(trsMatrix(this.primitiveEdit.startInstanceTransform!), f3(startPrim.t)));
                const planeNormal = this.cameraPlaneNormal();
                const t = dot3(sub3(planeOrigin, ray.origin), planeNormal) / dot3(ray.dir, planeNormal);
                const startRS = mat3(trsMatrix({ t: [0, 0, 0], r: startPrim.r, s: startPrim.s }));
                if (this.primitiveEdit.state === TransformationState.Rotating) {
                    const p = add3(ray.origin, scale3(ray.dir, t));
                    let startDir = sub3(this.primitiveEdit.startPlanePos, planeOrigin);
                    let currDir = sub3(p, planeOrigin);
                    if (len3(startDir) > 0 && len3(currDir) > 0) {
                        startDir = norm3(startDir);
                        currDir = norm3(currDir);
                        const deltaAngle = this.planeAngle(planeNormal, currDir, this.primitiveEdit.referencePlaneDir) - this.planeAngle(planeNormal, startDir, this.primitiveEdit.referencePlaneDir);
                        const localNormal = norm3(t3(transformVector(inverse(trsMatrix(this.primitiveEdit.startInstanceTransform!)), f3(planeNormal))));
                        const rotated = mulQuat(startPrim.r, quatFromAngleAxis(-deltaAngle, f3(localNormal)));
                        const m = mat3(trsMatrix({ t: [0, 0, 0], r: rotated, s: startPrim.s }));
                        this.currentEdit.primitive.invRotationScale = transpose3(inverse3(m));
                    }
                } else if (this.primitiveEdit.state === TransformationState.Scaling) {
                    const deltaX = this.ui.currentMousePosition[0] - this.primitiveEdit.startMousePos[0];
                    const scale = Math.pow(deltaX < 0 ? 1 / 1.05 : 1.05, (100 * Math.abs(deltaX)) / this.frameDim[0]);
                    if (this.primitiveEdit.axis === SDFEditorAxis.All) {
                        this.currentEdit.primitive.invRotationScale = transpose3(inverse3(mul3(startRS, [scale, 0, 0, 0, scale, 0, 0, 0, scale])));
                    } else if (this.primitiveEdit.axis === SDFEditorAxis.OpSmoothing) {
                        this.currentEdit.primitive.operationSmoothing = Math.min(Math.max(this.primitiveEdit.startPrimitive!.operationSmoothing * scale, kMinOpSmoothingRadius), kMaxOpSmoothingRadius);
                    } else {
                        const s = [1, 0, 0, 0, 1, 0, 0, 0, 1];
                        s[this.primitiveEdit.axis * 4] = scale;
                        this.currentEdit.primitive.invRotationScale = transpose3(inverse3(mul3(startRS, s)));
                    }
                }
                if (this.ui.symmetryPlane.active) this.updateSymmetryPrimitive();
            }
        }
    }

    private handleToggleSymmetryPlane(): void {
        this.ui.symmetryPlane.active = !this.ui.symmetryPlane.active;
        if (this.ui.symmetryPlane.active) this.updateSymmetryPrimitive();
        if (this.editingKeyDown) {
            const prims = this.gridPrimitives(true)!;
            if (this.ui.symmetryPlane.active) this.currentEdit.symmetryPrimitiveID = prims.addPrimitives([this.currentEdit.symmetryPrimitive]);
            else if (this.currentEdit.primitiveID !== kInvalidPrimitiveID) {
                prims.removePrimitives([this.currentEdit.symmetryPrimitiveID]);
                this.currentEdit.symmetryPrimitiveID = kInvalidPrimitiveID;
            }
        }
    }

    private primitivesAffected(keyPressedCount: number): number {
        return Math.min(Math.max(Math.floor(keyPressedCount / 7), 1), 10);
    }

    handleUndo(): void {
        const count = this.primitivesAffected(this.undoPressedCount++);
        if (this.nonBakedPrimitiveCount === 0) return;
        const n = Math.min(count, this.performedEdits.length, this.nonBakedPrimitiveCount);
        const toRemove: number[] = [];
        for (let i = 0; i < n; i++) toRemove.push(this.performedEdits.pop()!.primitiveID);
        const prims = this.gridPrimitives();
        if (!prims || toRemove.length === 0) return;
        for (const id of toRemove) {
            this.undoneEdits.push({ gridID: this.currentEdit.gridID, primitive: clonePrimitive(prims.getPrimitive(id)) });
            this.nonBakedPrimitiveCount--;
        }
        prims.removePrimitives(toRemove);
    }

    handleRedo(): void {
        const count = this.primitivesAffected(this.redoPressedCount++);
        const n = Math.min(count, this.undoneEdits.length);
        const toAdd: SDF3DPrimitive[] = [];
        for (let i = 0; i < n; i++) toAdd.push(this.undoneEdits.pop()!.primitive);
        if (toAdd.length === 0) return;
        const base = this.gridPrimitives(true)!.addPrimitives(toAdd);
        // Native counts the batch once (mNonBakedPrimitiveCount++ per grid).
        this.nonBakedPrimitiveCount++;
        for (let id = base; id < base + toAdd.length; id++) this.performedEdits.push({ gridID: this.currentEdit.gridID, primitiveID: id });
    }

    private bakePrimitives(): void {
        if (this.nonBakedPrimitiveCount > this.bakePrimitivesBatchSize + this.preservedHistoryCount) {
            const batches = Math.floor((this.nonBakedPrimitiveCount - this.preservedHistoryCount) / this.bakePrimitivesBatchSize);
            const count = this.bakePrimitivesBatchSize * batches;
            this.gridPrimitives()?.bakePrimitives(count);
            this.nonBakedPrimitiveCount -= count;
        }
    }

    private addEditPrimitive(addToCurrentEdit: boolean, addToHistory: boolean): void {
        const prims = this.gridPrimitives(true)!;
        if (this.ui.symmetryPlane.active) {
            this.updateSymmetryPrimitive();
            const base = prims.addPrimitives([this.currentEdit.primitive, this.currentEdit.symmetryPrimitive]);
            if (addToCurrentEdit) {
                this.currentEdit.primitiveID = base;
                this.currentEdit.symmetryPrimitiveID = base + 1;
            }
            if (addToHistory) this.performedEdits.push({ gridID: this.currentEdit.gridID, primitiveID: base }, { gridID: this.currentEdit.gridID, primitiveID: base + 1 });
            this.nonBakedPrimitiveCount += 2;
        } else {
            const base = prims.addPrimitives([this.currentEdit.primitive]);
            if (addToCurrentEdit) this.currentEdit.primitiveID = base;
            if (addToHistory) this.performedEdits.push({ gridID: this.currentEdit.gridID, primitiveID: base });
            this.nonBakedPrimitiveCount++;
        }
    }

    private removeEditPrimitives(): void {
        const prims = this.gridPrimitives();
        if (!prims) return;
        if (this.ui.symmetryPlane.active) {
            if (this.currentEdit.primitiveID !== kInvalidPrimitiveID && this.currentEdit.symmetryPrimitiveID !== kInvalidPrimitiveID) {
                prims.removePrimitives([this.currentEdit.primitiveID, this.currentEdit.symmetryPrimitiveID]);
                this.currentEdit.primitiveID = kInvalidPrimitiveID;
                this.currentEdit.symmetryPrimitiveID = kInvalidPrimitiveID;
                this.nonBakedPrimitiveCount -= 2;
            }
        } else if (this.currentEdit.primitiveID !== kInvalidPrimitiveID) {
            prims.removePrimitives([this.currentEdit.primitiveID]);
            this.currentEdit.primitiveID = kInvalidPrimitiveID;
            this.nonBakedPrimitiveCount--;
        }
    }

    private updateEditPrimitives(): void {
        const prims = this.gridPrimitives();
        if (!prims) return;
        prims.updatePrimitives([[this.currentEdit.primitiveID, this.currentEdit.primitive]]);
        if (this.ui.symmetryPlane.active) {
            this.updateSymmetryPrimitive();
            prims.updatePrimitives([[this.currentEdit.symmetryPrimitiveID, this.currentEdit.symmetryPrimitive]]);
        }
    }

    private handleToggleEditing(): void {
        if (this.editingKeyDown) {
            const p = this.handlePicking();
            if (!p) return;
            this.currentEdit.primitive.translation = p;
            this.addEditPrimitive(true, false);
        } else this.removeEditPrimitives();
    }

    private handleEditMovement(): void {
        const p = this.handlePicking();
        if (!p) return;
        this.currentEdit.primitive.translation = p;
        if (this.currentEdit.primitiveID !== kInvalidPrimitiveID) this.updateEditPrimitives();
    }

    private handleAddPrimitive(): void {
        const p = this.handlePicking();
        if (!p) return;
        this.currentEdit.primitive.translation = p;
        this.updateEditPrimitives();
        this.addEditPrimitive(false, true);
        if (this.autoBakingEnabled) this.bakePrimitives();
    }

    /** Mirrors handlePicking: the mouse ray's hit (grid plane or picked surface) in grid-local space. */
    private handlePicking(): V3 | null {
        if (!this.scene) return null;
        const { origin, dir } = this.mouseRay();
        let hit: V3;
        if (this.ui.gridPlane.active) {
            const pl = this.ui.gridPlane;
            const t = -dot3(pl.normal, sub3(origin, pl.position)) / dot3(pl.normal, dir);
            hit = add3(origin, scale3(dir, t));
        } else if (this.allowEditingOnOtherSurfaces || this.picking.instanceID === this.currentEdit.instanceID) {
            hit = add3(origin, scale3(dir, this.picking.distance));
        } else return null;
        return t3(transformPoint(inverse(this.gridTransform()), f3(hit)));
    }

    /** Mirrors onKeyEvent. */
    onKeyEvent(e: SDFEditorKeyEvent): boolean {
        if (!this.scene) return false;
        const keys = this.ui.keys;
        keys.prevShift = keys.shift;
        keys.prevControl = keys.control;
        const shift = e.mods?.shift ?? false;
        const ctrl = e.mods?.ctrl ?? false;
        const scene = this.scene;
        if (e.type === "keyPressed") {
            switch (e.key) {
                case "LeftControl":
                case "RightControl":
                    keys.control = true;
                    scene.setCameraControlsEnabled(false);
                    return true;
                case "LeftAlt":
                case "RightAlt":
                    this.editingKeyDown = true;
                    this.instanceEdit.state = TransformationState.None;
                    this.primitiveEdit.state = TransformationState.None;
                    scene.setCameraControlsEnabled(false);
                    this.handleToggleEditing();
                    return true;
                case "LeftShift":
                case "RightShift":
                    keys.shift = true;
                    this.primitiveEdit.state = TransformationState.None;
                    scene.setCameraControlsEnabled(false);
                    return true;
                case "Tab":
                    this.guiKeyDown = true;
                    this.ui.recordStartingMousePos = true;
                    this.instanceEdit.state = TransformationState.None;
                    scene.setCameraControlsEnabled(false);
                    return true;
                case "G":
                    this.ui.gridPlane.active = !this.ui.gridPlane.active;
                    return true;
                case "H":
                    this.handleToggleSymmetryPlane();
                    return true;
                case "B":
                    this.ui.bbRenderSettings.renderMode = (this.ui.bbRenderSettings.renderMode + 1) % SDFBBRenderMode.Count;
                    return true;
                case "C":
                    if (!this.editingKeyDown && !ctrl && !shift) {
                        this.allowEditingOnOtherSurfaces = !this.allowEditingOnOtherSurfaces;
                        return true;
                    }
                    break;
                case "X":
                    if (!this.editingKeyDown && !ctrl && !shift) {
                        this.previewEnabled = !this.previewEnabled;
                        return true;
                    }
                    break;
                case "Z":
                    if (ctrl && !this.editingKeyDown) {
                        this.undoPressedCount = 0;
                        keys.undo = true;
                        keys.redo = false;
                        this.handleUndo();
                        return true;
                    }
                    break;
                case "Y":
                    if (ctrl && !this.editingKeyDown) {
                        this.redoPressedCount = 0;
                        keys.redo = true;
                        keys.undo = false;
                        this.handleRedo();
                        return true;
                    }
                    break;
                case "T":
                    if (!this.editingKeyDown && shift) {
                        this.instanceEdit.state = TransformationState.Translating;
                        return true;
                    }
                    break;
                case "R":
                case "S": {
                    if (this.editingKeyDown) break;
                    const state = e.key === "R" ? TransformationState.Rotating : TransformationState.Scaling;
                    if (shift) {
                        this.instanceEdit.state = state;
                        return true;
                    }
                    if (ctrl) {
                        this.primitiveEdit.state = state;
                        return true;
                    }
                    break;
                }
                case "Key1":
                case "Key2":
                case "Key3":
                case "Key4":
                    if (this.primitiveEdit.state === TransformationState.Scaling) {
                        const axis = [SDFEditorAxis.X, SDFEditorAxis.Y, SDFEditorAxis.Z, SDFEditorAxis.OpSmoothing][Number(e.key.slice(3)) - 1]!;
                        this.primitiveEdit.axis = this.primitiveEdit.axis === axis ? SDFEditorAxis.All : axis;
                        return true;
                    }
                    break;
            }
        } else if (e.type === "keyReleased") {
            switch (e.key) {
                case "LeftControl":
                case "RightControl":
                    keys.control = false;
                    keys.undo = false;
                    keys.redo = false;
                    scene.setCameraControlsEnabled(true);
                    return true;
                case "LeftAlt":
                case "RightAlt":
                    this.editingKeyDown = false;
                    scene.setCameraControlsEnabled(true);
                    this.handleToggleEditing();
                    return true;
                case "LeftShift":
                case "RightShift":
                    keys.shift = false;
                    scene.setCameraControlsEnabled(true);
                    return true;
                case "Tab":
                    this.guiKeyDown = false;
                    this.ui.recordStartingMousePos = false;
                    if (!this.gridPlaneManipulated() && !this.symmetryPlaneManipulated() && this.primitiveEdit.state === TransformationState.None) {
                        this.ui.timeOfReleaseMainGUIKey = performance.now();
                        this.ui.fadeAwayGUI = true;
                    }
                    scene.setCameraControlsEnabled(true);
                    return true;
                case "Z":
                    keys.undo = false;
                    return true;
                case "Y":
                    keys.redo = false;
                    return true;
            }
        } else if (e.type === "keyRepeated") {
            if ((e.key === "Z" || e.key === "Y") && ctrl && !this.editingKeyDown) {
                if (e.key === "Z") this.handleUndo();
                else this.handleRedo();
                return true;
            }
        }
        return false;
    }

    /** Mirrors onMouseEvent. */
    onMouseEvent(e: SDFEditorMouseEvent): boolean {
        const pos: V2 = [e.pos[0] * this.frameDim[0], e.pos[1] * this.frameDim[1]];
        this.ui.currentMousePosition = pos;
        const down = e.type === "buttonDown";
        if (e.button === "left" && (e.type === "buttonDown" || e.type === "buttonUp")) this.lmbDown = down;
        if (e.button === "right" && (e.type === "buttonDown" || e.type === "buttonUp")) this.rmbDown = down;
        if (e.button === "middle" && (e.type === "buttonDown" || e.type === "buttonUp")) this.mmbDown = down;
        let handled = false;
        const manipulate = (plane: SDFGridPlane, previous: "previousGridPlane" | "previousSymmetryPlane", button: "right" | "middle") => {
            if (e.type === "buttonDown" && e.button === button) {
                this.ui.startMousePosition = pos;
                this.ui[previous] = copyPlane(plane);
            } else if (e.type === "move") {
                if (this.ui.keys.shift && !this.ui.keys.prevShift) {
                    this.ui.startMousePosition = [...pos];
                    this.ui[previous] = copyPlane(plane);
                } else if (!this.ui.keys.shift && this.ui.keys.prevShift) this.ui[previous] = copyPlane(plane);
                this.manipulateGridPlane(plane, this.ui[previous], this.ui.keys.shift, this.ui.keys.control);
            }
        };
        if (this.isMainGUIKeyDown()) {
            if (this.ui.recordStartingMousePos) {
                this.ui.recordStartingMousePos = false;
                this.ui.startMousePosition = pos;
            }
        } else if (this.editingKeyDown && !this.gridPlaneManipulated()) {
            if (!this.lmbDown) {
                if (e.type === "move") {
                    this.handleEditMovement();
                    handled = true;
                }
            } else if ((e.type === "buttonDown" && e.button === "left") || e.type === "move") {
                this.handleAddPrimitive();
                handled = true;
            }
        } else if (this.gridPlaneManipulated()) manipulate(this.ui.gridPlane, "previousGridPlane", "right");
        else if (this.symmetryPlaneManipulated()) manipulate(this.ui.symmetryPlane, "previousSymmetryPlane", "middle");

        if (this.lmbDown) {
            this.instanceEdit.state = TransformationState.None;
            this.primitiveEdit.state = TransformationState.None;
        } else if (e.type === "buttonUp" && e.button === "right") this.ui.previousGridPlane = copyPlane(this.ui.gridPlane);
        else if (e.type === "buttonUp" && e.button === "middle") this.ui.previousSymmetryPlane = copyPlane(this.ui.symmetryPlane);
        else if (e.type === "wheel") {
            const dy = e.wheelDelta?.[1] ?? 0;
            this.ui.scrollDelta += dy;
            if (this.instanceEdit.state !== TransformationState.None) this.instanceEdit.scrollTotal += dy;
        }
        return handled;
    }

    private manipulateGridPlane(plane: SDFGridPlane, previous: SDFGridPlane, translate: boolean, constrained: boolean): void {
        const diffPrev: V2 = [this.ui.currentMousePosition[0] - this.ui.prevMousePosition[0], this.ui.currentMousePosition[1] - this.ui.prevMousePosition[1]];
        const diffStart: V2 = [this.ui.currentMousePosition[0] - this.ui.startMousePosition[0], this.ui.currentMousePosition[1] - this.ui.startMousePosition[1]];
        const cam = this.scene!.camera;
        const up = t3(cam.getUpVector());
        const right = cross3(norm3(sub3(t3(cam.getTarget()), t3(cam.getPosition()))), up);
        const diagonal = Math.hypot(this.frameDim[0], this.frameDim[1]);
        const rotate = (mouseDiff: number, axis: V3, inN: V3, inR: V3, fromPrevious = true): [V3, V3] => {
            const maxAngle = Math.PI * 0.05;
            const angle = fromPrevious ? Math.min(Math.max(mouseDiff * Math.abs(mouseDiff) * ((2 * Math.PI * 0.075) / diagonal), -maxAngle), maxAngle) : mouseDiff * ((2 * Math.PI * 0.5) / diagonal);
            const m = rotateAround(angle, axis);
            return [norm3(t3(transformVector(m, f3(inN)))), norm3(t3(transformVector(m, f3(inR))))];
        };
        const move = (mouseDiff: number, axis: V3, from: V3): V3 => add3(from, scale3(axis, mouseDiff * Math.abs(mouseDiff) * (0.5 / diagonal)));
        if (!translate) {
            if (!constrained) {
                [plane.normal, plane.rightVector] = rotate(diffPrev[1], right, previous.normal, previous.rightVector);
                [plane.normal, plane.rightVector] = rotate(diffPrev[0], up, plane.normal, plane.rightVector);
                Object.assign(previous, copyPlane(plane));
            } else if (Math.abs(diffStart[1]) > Math.abs(diffStart[0])) [plane.normal, plane.rightVector] = rotate(diffStart[1], right, previous.normal, previous.rightVector, false);
            else [plane.normal, plane.rightVector] = rotate(diffStart[0], up, previous.normal, previous.rightVector, false);
        } else if (!constrained) {
            plane.position = move(diffPrev[0], right, previous.position);
            plane.position = move(-diffPrev[1], up, plane.position);
            Object.assign(previous, copyPlane(plane));
        } else if (Math.abs(diffStart[1]) > Math.abs(diffStart[0])) plane.position = move(-diffStart[1], up, previous.position);
        else plane.position = move(diffStart[0], right, previous.position);
    }

    override renderUI(ui: UIWidgets): void {
        ui.text("Help: Tab = shape/operation wheel, Alt = preview (Alt+LMB adds), Ctrl+R/S rotate/scale primitive, Shift+T/R/S transform instance, G/H/B/X/C toggles, Ctrl+Z/Y undo/redo.");
        ui.checkbox("Show/use grid plane", this.ui.gridPlane.active, (v) => (this.ui.gridPlane.active = v));
        if (this.ui.gridPlane.active) {
            ui.slider("Plane size", this.ui.gridPlane.planeSize, 0.01, 2, 0.01, (v) => (this.ui.gridPlane.planeSize = v));
            ui.slider("Grid line width", this.ui.gridPlane.gridLineWidth, 0.01, 0.1, 0.01, (v) => (this.ui.gridPlane.gridLineWidth = v));
            ui.slider("Grid scale", this.ui.gridPlane.gridScale, 0.01, 50, 0.01, (v) => (this.ui.gridPlane.gridScale = v));
        }
        ui.dropdown("Render Mode", ["Disabled", "Render All", "Render Selected"], ["Disabled", "Render All", "Render Selected"][this.ui.bbRenderSettings.renderMode]!, (v: string) => (this.ui.bbRenderSettings.renderMode = ["Disabled", "Render All", "Render Selected"].indexOf(v)));
        ui.slider("Edge Thickness", this.ui.bbRenderSettings.edgeThickness, 0.00001, 0.0005, 0.00001, (v) => (this.ui.bbRenderSettings.edgeThickness = v));
        ui.slider("Blobbing", this.ui.currentBlobbing, kMinShapeBlobbyness, kMaxShapeBlobbyness, 0.001, (v) => {
            this.ui.currentBlobbing = v;
            this.currentEdit.primitive.shapeBlobbing = v;
        });
        ui.checkbox("Auto baking", this.autoBakingEnabled, (v) => (this.autoBakingEnabled = v));
        const prims = this.gridPrimitives();
        ui.text(`#Primitives: ${prims?.primitiveCount ?? 0}, #Baked primitives: ${prims?.bakedPrimitiveCount ?? 0}`);
    }

    private ensureCopies(vbuffer: Texture, linearZ: Texture): void {
        const make = (src: Texture, format: ResourceFormat) =>
            new Texture(this.device, { type: ResourceType.Texture2D, width: src.width, height: src.height, format, mipLevels: 1, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess });
        if (!this.editingVBuffer || this.editingVBuffer.width !== vbuffer.width || this.editingVBuffer.height !== vbuffer.height) this.editingVBuffer = make(vbuffer, vbuffer.format);
        if (!this.editingLinearZ || this.editingLinearZ.width !== linearZ.width || this.editingLinearZ.height !== linearZ.height) this.editingLinearZ = make(linearZ, linearZ.format);
    }

    override execute(ctx: RenderContext, renderData: RenderData): void {
        const output = renderData.getTexture("output")!;
        const inputColor = renderData.getTexture("inputColor")!;
        if (!this.scene || !this.currentEdit.grid) {
            ctx.blit(inputColor, output);
            return;
        }
        this.frameDim = [output.width, output.height];
        let vbuffer = renderData.getTexture("vbuffer")!;
        const linearZ = renderData.getTexture("linearZ")!;
        this.ensureCopies(vbuffer, linearZ);
        // While editing, the copies from before the edit keep new primitives from picking themselves.
        if (!this.editingKeyDown) {
            ctx.copyTexture(this.editingVBuffer!, vbuffer);
            ctx.copyTexture(this.editingLinearZ!, linearZ);
        } else vbuffer = this.editingVBuffer!;

        this.setup2DGUI();
        this.handleActions();
        // Grid edits re-bake before the frame is drawn (native SDFGrid::update in Scene::update).
        this.scene.updateSDFGrids();

        if (!this.guiPass) this.guiPass = FullScreenPass.create(this.device, { path: kGUIPassShaderFilename, psEntry: "psMain", defines: this.scene.getSceneDefines() });
        this.fbo.attachColorTarget(output, 0);
        this.bindShaderData(this.guiPass.getRootVar(), inputColor, vbuffer);
        this.guiPass.execute(ctx, this.fbo);
        this.readPicking(ctx);

        // Prepare the next frame.
        this.ui.keys.prevShift = this.ui.keys.shift;
        this.ui.keys.prevControl = this.ui.keys.control;
        this.ui.prevMousePosition = [...this.ui.currentMousePosition];
        this.ui.scrollDelta = 0;
        this.instanceEdit.prevState = this.instanceEdit.state;
        this.instanceEdit.prevScrollTotal = this.instanceEdit.scrollTotal;
        this.primitiveEdit.prevState = this.primitiveEdit.state;
        this.primitiveEdit.prevAxis = this.primitiveEdit.axis;
    }

    /** The picking info written this frame, read back without stalling (§9: one frame late at worst). */
    private readPicking(ctx: RenderContext): void {
        if (this.pickingReadbackPending) return;
        this.pickingReadbackPending = true;
        void ctx
            .readBuffer(this.pickingInfo)
            .then((bytes: Uint8Array) => {
                const v = new DataView(bytes.buffer, bytes.byteOffset, 12);
                this.picking = { distance: v.getFloat32(0, true), instanceID: v.getUint32(4, true), hitType: v.getUint32(8, true) };
            })
            .catch((e: unknown) => Logger.warning(`SDFEditor: picking readback failed: ${String(e)}`))
            .finally(() => (this.pickingReadbackPending = false));
    }

    private bindShaderData(root: ShaderVar, inputColor: Texture, vbuffer: Texture): void {
        const scene = this.scene!;
        const data = new DataView(new ArrayBuffer(kEditingDataSize));
        data.setUint32(0, this.editingKeyDown ? 1 : 0, true);
        data.setUint32(4, this.previewEnabled ? 1 : 0, true);
        data.setUint32(8, this.currentEdit.instanceID, true);
        data.setUint32(12, this.primitiveEdit.state !== TransformationState.Scaling ? SDFEditorAxis.Count : this.primitiveEdit.axis, true);
        packPrimitive(data, 16, this.currentEdit.primitive);
        packPrimitive(data, 128, this.currentEdit.symmetryPrimitive);
        packAABB(data, 240, computePrimitiveAABB(this.currentEdit.primitive));
        packAABB(data, 272, computePrimitiveAABB(this.currentEdit.symmetryPrimitive));
        this.editingDataBuffer.setBlob(new Uint8Array(data.buffer));

        const ids = scene.getSDFGridInstanceIDs();
        if (!this.gridInstanceIDsBuffer || this.gridInstanceCount < ids.length) {
            this.gridInstanceCount = ids.length;
            this.gridInstanceIDsBuffer = new Buffer(this.device, { size: Math.max(1, ids.length) * 4, structSize: 4, bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess, memoryType: MemoryType.DeviceLocal, name: "SDFEditor::gridInstanceIDs" });
            this.gridInstanceIDsBuffer.setBlob(new Uint32Array(ids));
        }

        root["gInputColor"] = inputColor;
        root["gLinearZ"] = this.editingLinearZ!;
        root["gVBuffer"] = vbuffer;
        scene.bindShaderData(root);
        const g = root["gGUIPass"] as ShaderVar;
        g["resolution"] = this.frameDim;
        g["mousePos"] = [Math.trunc(this.ui.currentMousePosition[0]), Math.trunc(this.ui.currentMousePosition[1])];
        g["pickingData"] = this.pickingInfo;
        g["editingPrimitiveData"] = this.editingDataBuffer;
        g["gridInstanceIDs"] = this.gridInstanceIDsBuffer;
        const bb = g["bbRenderSettings"] as ShaderVar;
        bb["renderMode"] = this.ui.bbRenderSettings.renderMode;
        bb["selectedInstanceID"] = this.ui.bbRenderSettings.selectedInstanceID;
        bb["edgeThickness"] = this.ui.bbRenderSettings.edgeThickness;
        g["gridInstanceCount"] = this.gridInstanceCount;
        g["ui2DActive"] = this.isMainGUIKeyDown() ? 1 : 0;
        const plane = (v: ShaderVar, p: SDFGridPlane) => {
            v["position"] = p.position;
            v["gridLineWidth"] = p.gridLineWidth;
            v["normal"] = p.normal;
            v["gridScale"] = p.gridScale;
            v["rightVector"] = p.rightVector;
            v["planeSize"] = p.planeSize;
            v["color"] = p.color;
            v["active"] = p.active ? 1 : 0;
        };
        plane(g["gridPlane"] as ShaderVar, this.ui.gridPlane);
        plane(g["symmetryPlane"] as ShaderVar, this.ui.symmetryPlane);
        this.markers.bindShaderData(g["markerSet"] as ShaderVar);
    }
}

registerRenderPass("SDFEditor", (device, props) => new SDFEditor(device, props));

