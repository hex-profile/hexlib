#pragma once

#include "data/array.h"
#include "data/gpuPtr.h"

//================================================================
//
// GpuArray<Type>
//
// Array for GPU address space.
//
//================================================================

template <typename Type>
using GpuArray = ArrayEx<GpuPtr(Type)>;

//================================================================
//
// makeConst (fast)
//
//================================================================

#if GpuPtrDistinctType

template <typename Type>
sysinline const GpuArray<const Type>& makeConst(const GpuArray<Type>& array)
{
    return recastEqualLayout<const GpuArray<const Type>>(array);
}

#endif

//================================================================
//
// recastElement
//
// Use with caution!
//
//================================================================

#if GpuPtrDistinctType

template <typename Dst, typename Src>
sysinline auto& recastElement(const GpuArray<Src>& array)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(Src, Dst);
    return recastEqualLayout<const GpuArray<Dst>>(array);
}

#endif
