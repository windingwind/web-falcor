/**
 * WebFalcor override of Utils/Sampling/Pseudorandom/SplitMix64.slang.
 * Diff vs upstream: WGSL has no 64-bit integers, so the uint64_t state is
 * emulated as uint2 (x = low, y = high 32 bits) with explicit carry/mul
 * emulation. nextRandom64 returns uint2 instead of uint64_t; sequences are
 * bit-identical to upstream (verified against the reference implementation).
 */
/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * SplitMix64 pseudorandom number generator.
 *
 * This is a fixed-increment version of Java 8's SplittableRandom generator.
 * The period is 2^64 and its state size is 64 bits.
 *
 * WebFalcor: 64-bit state emulated as uint2 (x = low, y = high).
 */

struct SplitMix64
{
    uint2 state;
};

/** 64-bit add on (lo, hi) pairs. */
uint2 falcor_u64_add(uint2 a, uint2 b)
{
    uint lo = a.x + b.x;
    uint carry = lo < a.x ? 1u : 0u;
    return uint2(lo, a.y + b.y + carry);
}

/** 64-bit logical right shift, 1 <= k < 32. */
uint2 falcor_u64_shr(uint2 a, uint k)
{
    return uint2((a.x >> k) | (a.y << (32u - k)), a.y >> k);
}

/** Full 32x32 -> 64 bit multiply. */
uint2 falcor_mul32x32(uint a, uint b)
{
    uint a0 = a & 0xffffu, a1 = a >> 16;
    uint b0 = b & 0xffffu, b1 = b >> 16;
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint mid = p01 + p10;                       // may wrap: contributes 2^48
    uint midCarry = mid < p01 ? 0x10000u : 0u;
    uint lo = p00 + (mid << 16);
    uint loCarry = lo < p00 ? 1u : 0u;
    uint hi = p11 + (mid >> 16) + midCarry + loCarry;
    return uint2(lo, hi);
}

/** Low 64 bits of a 64x64 multiply. */
uint2 falcor_u64_mul(uint2 a, uint2 b)
{
    uint2 r = falcor_mul32x32(a.x, b.x);
    r.y += a.x * b.y + a.y * b.x;
    return r;
}

/**
 * Generates the next pseudorandom number in the sequence (64 bits as (lo, hi)).
 */
uint2 nextRandom64(inout SplitMix64 rng)
{
    rng.state = falcor_u64_add(rng.state, uint2(0x7F4A7C15u, 0x9E3779B9u)); // += 0x9E3779B97F4A7C15ull
    uint2 z = rng.state;
    z = falcor_u64_mul(z ^ falcor_u64_shr(z, 30), uint2(0x1CE4E5B9u, 0xBF58476Du)); // * 0xBF58476D1CE4E5B9ull
    z = falcor_u64_mul(z ^ falcor_u64_shr(z, 27), uint2(0x133111EBu, 0x94D049BBu)); // * 0x94D049BB133111EBull
    return z ^ falcor_u64_shr(z, 31);
}

/**
 * Generates the next pseudorandom number in the sequence (low 32 bits).
 */
uint nextRandom(inout SplitMix64 rng)
{
    return nextRandom64(rng).x;
}

/**
 * Initialize SplitMix64 pseudorandom number generator.
 * @param[in] s0 Low bits of initial state (seed).
 * @param[in] s1 High bits of initial state (seed).
 */
SplitMix64 createSplitMix64(uint s0, uint s1)
{
    SplitMix64 rng;
    rng.state = uint2(s0, s1);
    return rng;
}
