/**
 * WebFalcor override of Tests/Slang/SlangTests.cs.slang, reduced to testEnum,
 * testDefaultInitializers (without its double case) and testHashedStrings: the
 * other entry points use 16/64-bit scalar types (double, uint64_t, float16_t
 * storage) that WGSL lacks, and fail the whole module.
 */
/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
import SlangShared;

RWStructuredBuffer<uint> result;

[numthreads(1, 1, 1)]
void testEnum()
{
    // Store the enum values cast to uint.
    result[0] = (uint)Type1::A;
    result[1] = (uint)Type1::B;
    result[2] = (uint)Type1::C;
    result[3] = (uint)Type1::D;

    result[4] = (uint)Type2::A;
    result[5] = (uint)Type2::B;
    result[6] = (uint)Type2::C;
    result[7] = (uint)Type2::D;

    result[8] = (uint)Type3::A;
    result[9] = (uint)Type3::B;
    result[10] = (uint)Type3::C;
    result[11] = (uint)Type3::D;
}

struct A
{
    uint a;
    float b;
};

struct Foo
{
    int a;
    uint b;
    float c;
    bool d;
    float2x2 e;
    A f;
};

struct Foo2
{
    int a;
    Type3 t = Type3::C;
};

[numthreads(1, 1, 1)]
void testDefaultInitializers()
{
    int i = 0;

    // Scalar types (i=0)
    {
        int a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = a;
        result[i++] = b;
        result[i++] = c;
    }
    {
        uint a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = a;
        result[i++] = b;
        result[i++] = c;
    }
    {
        float a = 1, b = 2, c = {};
        a = {};
        b = c;
        result[i++] = asuint(a);
        result[i++] = asuint(b);
        result[i++] = asuint(c);
    }
    // WebFalcor: the double case (i = 9..14) is skipped, WGSL has no 64-bit floats.
    i += 6;

    // Vector/matrix types (i=15)
    {
        float4 a = float4(1, 2, 3, 4), b = {};
        a = {};
        result[i++] = asuint(a.x);
        result[i++] = asuint(a.y);
        result[i++] = asuint(a.z);
        result[i++] = asuint(a.w);
        result[i++] = asuint(b.x);
        result[i++] = asuint(b.y);
        result[i++] = asuint(b.z);
        result[i++] = asuint(b.w);
    }
    {
        float2x2 a = float2x2(1, 2, 3, 4), b = {};
        a = {};
        result[i++] = asuint(a[0][0]);
        result[i++] = asuint(a[0][1]);
        result[i++] = asuint(a[1][0]);
        result[i++] = asuint(a[1][1]);
        result[i++] = asuint(b[0][0]);
        result[i++] = asuint(b[0][1]);
        result[i++] = asuint(b[1][0]);
        result[i++] = asuint(b[1][1]);
    }

    // Struct (i=31)
    {
        A myA;
        myA.a = 1;
        myA.b = 2;
        Foo foo;
        foo.a = 1;
        foo.b = 2;
        foo.c = 3;
        foo.d = true;
        foo.e = float2x2(1, 2, 3, 4);
        foo.f = myA;

        foo = {};

        result[i++] = asuint(foo.a);
        result[i++] = asuint(foo.b);
        result[i++] = asuint(foo.c);
        result[i++] = asuint(foo.d);
        result[i++] = asuint(foo.e[0][0]);
        result[i++] = asuint(foo.e[0][1]);
        result[i++] = asuint(foo.e[1][0]);
        result[i++] = asuint(foo.e[1][1]);
        result[i++] = asuint(foo.f.a);
        result[i++] = asuint(foo.f.b);
    }

    // Struct with enum (i=41)
    {
        Foo2 foo;
        foo.a = 1;
        foo.t = Type3::B;

        foo = {};

        result[i++] = (uint)foo.a;
        result[i++] = (uint)foo.t; // Enum should be initialized to its default value specified in the struct declaration (i=36)
    }

    // All done (i=43)
}

[numthreads(1, 1, 1)]
void testHashedStrings()
{
    result[0] = getStringHash("Test String 0");
    result[1] = getStringHash("Test String 1");
    result[2] = getStringHash("Test String 2");
    result[3] = getStringHash("Test String 3");
}
