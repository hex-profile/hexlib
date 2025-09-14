#pragma once

#if !defined(__CUDA_ARCH__)
    #include <atomic>
#endif

#include "compileTools/compileTools.h"
#include "numbers/float/floatBase.h"
#include "numbers/int/intBase.h"

//================================================================
//
// atomicAction
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

template <typename Type, typename Action>
sysinline Type atomicAction(Type* dst, Type value, const Action& action)
{
    using namespace std;

    using AtomicType = atomic<Type>;
    COMPILE_ASSERT(sizeof(AtomicType) == sizeof(Type));
    auto& target = * (AtomicType*) dst;

    auto oldValue = target.load();

    while (!target.compare_exchange_weak(oldValue, action(oldValue, value), memory_order_relaxed, memory_order_relaxed))
        ;

    return oldValue;
}

#endif


//================================================================
//
// atomicAdd
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

template <typename Type>
sysinline Type atomicAdd(Type* dst, Type value)
{
    return atomicAction(dst, value, [] (auto a, auto b) {return a + b;});
}

#endif

//================================================================
//
// atomicMin
// atomicMax
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

//----------------------------------------------------------------

template <typename Type>
sysinline Type atomicMin(Type* dst, Type value)
{
    return atomicAction(dst, value, [] (auto a, auto b) {return minv(a, b);});
}

template <typename Type>
sysinline Type atomicMax(Type* dst, Type value)
{
    return atomicAction(dst, value, [] (auto a, auto b) {return maxv(a, b);});
}

//----------------------------------------------------------------

#endif

//================================================================
//
// atomicMaxNonneg
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

    sysinline float32 atomicMinNonneg(float32* dst, float32 value)
        {return atomicMin(dst, value);}

    sysinline float32 atomicMaxNonneg(float32* dst, float32 value)
        {return atomicMax(dst, value);}

#else

    sysinline float32 atomicMinNonneg(float32* dst, float32 value)
    {
        int32* dstInt = &recastEqualLayout<int32>(*dst);
        return atomicMin(dstInt, __float_as_int(value));
    }

    sysinline float32 atomicMaxNonneg(float32* dst, float32 value)
    {
        int32* dstInt = &recastEqualLayout<int32>(*dst);
        return atomicMax(dstInt, __float_as_int(value));
    }

#endif

//================================================================
//
// __float_as_int
// __int_as_float
//
// For host.
//
//================================================================

#if !defined(__CUDA_ARCH__)

sysinline int32 __float_as_int(float32 value)
{
    return recastEqualLayout<int32>(value);
}

sysinline float32 __int_as_float(int32 value)
{
    return recastEqualLayout<float32>(value);
}

#endif

//================================================================
//
// orderedIntFromFloat
// orderedIntToFloat
//
// Originally posted by Andy_Lomas on NVIDIA forums.
//
// Tested by me, correctly handles both +-{0, 1, maxfloat, infinity},
// including negative zero.
//
//================================================================

template <typename Type>
sysinline int32 orderedIntFromFloat(Type value)
    MISSING_FUNCTION_BODY

template <>
sysinline int32 orderedIntFromFloat(float32 value)
{
    int32 intVal = __float_as_int(value);
    if_not (intVal >= 0) intVal ^= 0x7FFFFFFF;
    return intVal;
}

//----------------------------------------------------------------

template <typename Type>
sysinline float32 orderedIntToFloat(Type value)
    MISSING_FUNCTION_BODY

template <>
sysinline float32 orderedIntToFloat(int32 value)
{
    int32 intVal = value;
    if_not (intVal >= 0) intVal ^= 0x7FFFFFFF;
    return __int_as_float(intVal);
}
