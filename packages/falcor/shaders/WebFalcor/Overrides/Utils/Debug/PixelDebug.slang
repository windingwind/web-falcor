/**
 * WebFalcor override of Utils/Debug/PixelDebug.slang (docs §9). Same public
 * API (printSetPixel / print / assert); the ParameterBlock<PixelDebug> with
 * RWByteAddressBuffer counters becomes global buffers: an Atomic counter
 * pair plus one flat uint record buffer (8-uint print records followed by
 * 4-uint assert records).
 */

import Utils.Debug.PixelDebugTypes;

#ifdef _PIXEL_DEBUG_ENABLED
cbuffer PixelDebugCB
{
    uint2 gPixelDebugSelected;
    uint gPixelDebugPrintCapacity;
    uint gPixelDebugAssertCapacity;
}

RWStructuredBuffer<Atomic<uint>> gPixelDebugCounters; ///< [0] prints, [1] asserts.
RWStructuredBuffer<uint> gPixelDebugRecords; ///< Print records, then assert records.
#endif

static uint2 gPixelDebugPixel;

void printSetPixel(uint2 pixel)
{
#ifdef _PIXEL_DEBUG_ENABLED
    gPixelDebugPixel = pixel;
#endif
}

/// Interface for values that can be printed (mirrors upstream).
interface IPrintable
{
    uint getPrintableDataVal();
    static PrintValueType getPrintableValueType();
}

extension bool : IPrintable
{
    uint getPrintableDataVal() { return uint(this); }
    static PrintValueType getPrintableValueType() { return PrintValueType::Bool; }
}

extension int : IPrintable
{
    uint getPrintableDataVal() { return asuint(this); }
    static PrintValueType getPrintableValueType() { return PrintValueType::Int; }
}

extension uint : IPrintable
{
    uint getPrintableDataVal() { return this; }
    static PrintValueType getPrintableValueType() { return PrintValueType::Uint; }
}

extension float : IPrintable
{
    uint getPrintableDataVal() { return asuint(this); }
    static PrintValueType getPrintableValueType() { return PrintValueType::Float; }
}

#ifdef _PIXEL_DEBUG_ENABLED

[ForceInline]
void __printRecord(String msg, PrintValueType valueType, int count, uint4 data)
{
    if (all(gPixelDebugPixel == gPixelDebugSelected))
    {
        uint index = gPixelDebugCounters[0].add(1);
        if (index < gPixelDebugPrintCapacity)
        {
            uint base = index * 8;
            gPixelDebugRecords[base + 0] = getStringHash(msg);
            gPixelDebugRecords[base + 1] = (uint)valueType;
            gPixelDebugRecords[base + 2] = (uint)count;
            gPixelDebugRecords[base + 3] = 0;
            gPixelDebugRecords[base + 4] = data.x;
            gPixelDebugRecords[base + 5] = data.y;
            gPixelDebugRecords[base + 6] = data.z;
            gPixelDebugRecords[base + 7] = data.w;
        }
    }
}

[ForceInline]
void print<T : IPrintable>(String msg, T v)
{
    uint4 u = uint4(0);
    u[0] = v.getPrintableDataVal();
    __printRecord(msg, T.getPrintableValueType(), 1, u);
}

[ForceInline]
void print<T : IPrintable, let N : int>(String msg, vector<T, N> v)
{
    uint4 u = uint4(0);
    [ForceUnroll]
    for (int i = 0; i < N; i++) u[i] = v[i].getPrintableDataVal();
    __printRecord(msg, T.getPrintableValueType(), N, u);
}

void print<T : IPrintable>(T v) { print("", v); }
void print<T : IPrintable, let N : int>(vector<T, N> v) { print("", v); }

void assert(bool condition, String msg = "")
{
    if (!condition)
    {
        uint index = gPixelDebugCounters[1].add(1);
        if (index < gPixelDebugAssertCapacity)
        {
            uint base = gPixelDebugPrintCapacity * 8 + index * 4;
            gPixelDebugRecords[base + 0] = gPixelDebugPixel.x;
            gPixelDebugRecords[base + 1] = gPixelDebugPixel.y;
            gPixelDebugRecords[base + 2] = 0;
            gPixelDebugRecords[base + 3] = getStringHash(msg);
        }
    }
}

#else // !_PIXEL_DEBUG_ENABLED

void print<T : IPrintable>(String msg, T v) {}
void print<T : IPrintable, let N : int>(String msg, vector<T, N> v) {}
void print<T : IPrintable>(T v) {}
void print<T : IPrintable, let N : int>(vector<T, N> v) {}
void assert(bool condition, String msg = "") {}

#endif
