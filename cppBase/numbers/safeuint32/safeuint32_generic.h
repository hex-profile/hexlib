#pragma once

#if defined(_MSC_VER)
    #define _DO_NOT_DECLARE_INTERLOCKED_INTRINSICS_IN_MEMORY
    #include <intrin.h>
    #define EMUL_AVAILABLE
#endif

#include "numbers/interface/numberInterface.h"
#include "numbers/int/intType.h"

//================================================================
//
// SafeBinaryImpl<uint32, OpAdd>
//
//================================================================

template <>
struct SafeBinaryImpl<uint32, OpAdd>
{
    struct Code
    {
        static sysinline bool func(uint32 X, uint32 Y, uint32& R)
        {
            R = X + Y;
            return X <= TYPE_MAX(uint32) - Y;
        }
    };
};

//================================================================
//
// SafeBinaryImpl<uint32, OpSub>
//
//================================================================

template <>
struct SafeBinaryImpl<uint32, OpSub>
{
    struct Code
    {
        static sysinline bool func(uint32 X, uint32 Y, uint32& R)
        {
            R = X - Y;
            return X >= Y;
        }
    };
};

//================================================================
//
// SafeBinaryImpl<uint32, OpMul>
//
//================================================================

template <>
struct SafeBinaryImpl<uint32, OpMul>
{
    struct Code
    {
        static sysinline bool func(uint32 X, uint32 Y, uint32& result)
        {
        #ifdef EMUL_AVAILABLE
            uint64 res64 = __emulu(X, Y);
        #elif defined(__i386__) || defined(__x86_64__) || defined(__arm__) || defined(__aarch64__)
            uint64 res64 = uint64(X) * uint64(Y);
        #else
            #define Need to implement
        #endif

            uint32 resHi = res64 >> 32;
            result = uint32(res64);
            return (resHi == 0);
        }
    };
};
