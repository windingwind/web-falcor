/**
 * Host side of Utils/Debug/PixelDebug: captures shader print()/assert()
 * records for one selected pixel. Web divergences (docs §9): records read
 * back asynchronously (~1 frame late) instead of native's fence wait, and
 * message strings surface as slang string hashes (hashed-string reflection
 * is not exposed by slang-wasm); passes must compile with the
 * _PIXEL_DEBUG_ENABLED define (see getDefines()).
 */

import { Buffer } from "../../Core/API/Buffer.js";
import type { ComputeContext } from "../../Core/API/ComputeContext.js";
import type { Device } from "../../Core/API/Device.js";
import { MemoryType, ResourceBindFlags } from "../../Core/API/Types.js";
import type { ShaderVar } from "../../Core/Program/ParameterBlock.js";

export enum PrintValueType { Bool = 0, Int, Uint, Float }

export interface PrintRecord {
    msgHash: number;
    type: PrintValueType;
    /** Decoded component values (1-4). */
    values: (number | boolean)[];
}

export interface AssertRecord {
    launchX: number;
    launchY: number;
    msgHash: number;
}

const kPrintCapacity = 100;
const kAssertCapacity = 100;

export class PixelDebug {
    private countersBuffer: Buffer | null = null;
    private recordsBuffer: Buffer | null = null;
    private readbackInFlight = false;
    private prints: PrintRecord[] = [];
    private asserts: AssertRecord[] = [];

    enabled = false;
    selectedPixel: [number, number] = [0, 0];

    constructor(private readonly device: Device) {}

    /** Compile-time define a debugged program must include. */
    static getDefines(): Record<string, string> {
        return { _PIXEL_DEBUG_ENABLED: "1" };
    }

    /** Mirrors PixelDebug::beginFrame: clears the per-frame counters. */
    beginFrame(ctx: ComputeContext): void {
        if (!this.enabled) return;
        this.countersBuffer ??= new Buffer(this.device, {
            size: 2 * 4,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "PixelDebug::counters",
        });
        this.recordsBuffer ??= new Buffer(this.device, {
            size: (kPrintCapacity * 8 + kAssertCapacity * 4) * 4,
            structSize: 4,
            bindFlags: ResourceBindFlags.ShaderResource | ResourceBindFlags.UnorderedAccess,
            memoryType: MemoryType.DeviceLocal,
            name: "PixelDebug::records",
        });
        ctx.clearBuffer(this.countersBuffer);
    }

    /** Mirrors PixelDebug::prepareProgram: binds the debug resources
     *  (members absent when the define is off or DCE removed them). */
    prepareProgram(root: ShaderVar): void {
        if (!this.enabled) return;
        const trySet = (name: string, value: unknown) => {
            try {
                (root as Record<string, unknown>)[name] = value;
            } catch {
                /* absent in this variant */
            }
        };
        try {
            const cb = root["PixelDebugCB"] as ShaderVar;
            cb["gPixelDebugSelected"] = this.selectedPixel;
            cb["gPixelDebugPrintCapacity"] = kPrintCapacity;
            cb["gPixelDebugAssertCapacity"] = kAssertCapacity;
        } catch {
            /* debug cbuffer absent */
        }
        trySet("gPixelDebugCounters", this.countersBuffer);
        trySet("gPixelDebugRecords", this.recordsBuffer);
    }

    /** Mirrors PixelDebug::endFrame: starts the async record readback. */
    endFrame(): void {
        if (!this.enabled || this.readbackInFlight || !this.countersBuffer) return;
        this.readbackInFlight = true;
        void Promise.all([this.countersBuffer.getBlob(), this.recordsBuffer!.getBlob()]).then(([counters, records]) => {
            const counts = new Uint32Array(counters.buffer, counters.byteOffset, 2);
            const u32 = new Uint32Array(records.buffer, records.byteOffset, records.byteLength / 4);
            const f32 = new Float32Array(u32.buffer, u32.byteOffset, u32.length);
            const i32 = new Int32Array(u32.buffer, u32.byteOffset, u32.length);

            const prints: PrintRecord[] = [];
            for (let r = 0; r < Math.min(counts[0]!, kPrintCapacity); r++) {
                const base = r * 8;
                const type = u32[base + 1]! as PrintValueType;
                const count = u32[base + 2]!;
                const values: (number | boolean)[] = [];
                for (let c = 0; c < Math.min(count, 4); c++) {
                    const bits = base + 4 + c;
                    if (type === PrintValueType.Float) values.push(f32[bits]!);
                    else if (type === PrintValueType.Int) values.push(i32[bits]!);
                    else if (type === PrintValueType.Bool) values.push(u32[bits]! !== 0);
                    else values.push(u32[bits]!);
                }
                prints.push({ msgHash: u32[base]!, type, values });
            }
            const asserts: AssertRecord[] = [];
            for (let r = 0; r < Math.min(counts[1]!, kAssertCapacity); r++) {
                const base = kPrintCapacity * 8 + r * 4;
                asserts.push({ launchX: u32[base]!, launchY: u32[base + 1]!, msgHash: u32[base + 3]! });
            }
            this.prints = prints;
            this.asserts = asserts;
            this.readbackInFlight = false;
        });
    }

    /** Print records from the most recent resolved frame. */
    getPrintRecords(): PrintRecord[] {
        return this.prints;
    }

    /** Assert records from the most recent resolved frame. */
    getAssertRecords(): AssertRecord[] {
        return this.asserts;
    }
}
