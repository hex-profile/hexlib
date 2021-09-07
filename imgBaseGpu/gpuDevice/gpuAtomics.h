#pragma once

#if !defined(__CUDA_ARCH__)
    #include <atomic>
#endif

#include "compileTools/compileTools.h"

//================================================================
//
// atomicAdd
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

template <typename Type>
sysinline void atomicAdd(Type* dst, Type value)
{
    using namespace std;

    using AtomicType = atomic<Type>;
    COMPILE_ASSERT(sizeof(AtomicType) == sizeof(Type));
    auto& target = * (AtomicType*) dst;

    auto oldValue = target.load();

    while (!target.compare_exchange_weak(oldValue, oldValue + value, memory_order_relaxed, memory_order_relaxed)) 
        ;
}

#endif

//================================================================
//
// atomicMax
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

template <typename Type>
sysinline void atomicMax(Type* dst, Type value)
{
    using namespace std;

    using AtomicType = atomic<Type>;
    COMPILE_ASSERT(sizeof(AtomicType) == sizeof(Type));
    auto& target = * (AtomicType*) dst;

    ////

    auto oldValue = target.load();

    while (!target.compare_exchange_weak(oldValue, maxv(oldValue, value), memory_order_relaxed, memory_order_relaxed)) 
        ;
}

#endif

//================================================================
//
// atomicMaxNonneg
//
// Slow, but correct for x86.
//
//================================================================

#if !defined(__CUDA_ARCH__)

    sysinline void atomicMaxNonneg(float32* dst, float32 value)
    {
        return atomicMax(dst, value);
    }

#else

    sysinline void atomicMaxNonneg(float32* dst, float32 value)
    {
        int32* dstInt = &recastEqualLayout<int32>(*dst);
        atomicMax(dstInt, __float_as_int(value));
    }

#endif
