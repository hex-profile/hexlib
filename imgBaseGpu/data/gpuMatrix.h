#pragma once

#include "data/matrix.h"
#include "data/gpuPtr.h"

//================================================================
//
// GpuMatrix
//
// Matrix for GPU address space.
//
//================================================================

template <typename Type, typename Pitch = PitchDefault>
using GpuMatrix = MatrixEx<GpuPtr(Type), Pitch>;

//================================================================
//
// Verify memory layout to be identical for CPU and GPU.
//
//================================================================

COMPILE_ASSERT(sizeof(GpuMatrix<uint8>) == (HEXLIB_GPU_BITNESS == 32 ? 16 : 24));
COMPILE_ASSERT(alignof(GpuMatrix<uint8>) == HEXLIB_GPU_BITNESS / 8);

//================================================================
//
// GpuMatrixAP
//
//================================================================

template <typename Type>
using GpuMatrixAP = GpuMatrix<Type, PitchMayBeNegative>;

//================================================================
//
// makeConst (fast)
//
//================================================================

#if GpuPtrDistinctType

template <typename Type, typename Pitch>
sysinline auto& makeConst(const GpuMatrix<Type, Pitch>& value)
{
    return recastEqualLayout<const GpuMatrix<const Type, Pitch>>(value);
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

template <typename DstType, typename SrcType, typename SrcPitch>
sysinline auto& recastElement(const GpuMatrix<SrcType, SrcPitch>& matrix)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(SrcType, DstType);
    return recastEqualLayout<const GpuMatrix<DstType, SrcPitch>>(matrix);
}

#endif
