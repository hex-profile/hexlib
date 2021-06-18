#pragma once

#if !defined(__CUDA_ARCH__)
    #include <atomic>
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
inline void atomicAdd(Type* dst, Type value)
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
inline void atomicMax(Type* dst, Type value)
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
