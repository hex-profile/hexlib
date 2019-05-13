#pragma once

#include "data/matrix.h"
#include "data/gpuPtr.h"
#include "data/gpuArray.h"

//================================================================
//
// GpuMatrix
//
// Matrix for GPU address space: identical to MatrixEx<GpuPtr(Type)>.
//
//================================================================

template <typename Type>
class GpuMatrix : public MatrixEx<GpuPtr(Type)>
{

public:

    using Base = MatrixEx<GpuPtr(Type)>;
    using TmpType = typename Base::TmpType;

    //
    // Constructors
    //

    sysinline GpuMatrix(const TmpType* = 0)
        {}

    sysinline GpuMatrix(GpuPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

    sysinline GpuMatrix(GpuPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions& preconditions)
        : Base(memPtr, memPitch, sizeX, sizeY, preconditions) {}

#if HEXLIB_GUARDED_MEMORY

    sysinline GpuMatrix(ArrayPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

    sysinline GpuMatrix(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

    sysinline GpuMatrix(MatrixPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions& preconditions)
        : Base(memPtr, memPitch, sizeX, sizeY, preconditions) {}

#endif

    sysinline GpuMatrix(const Base& base)
        : Base(base) {}

    template <typename OtherPointer>
    sysinline GpuMatrix(const ArrayEx<OtherPointer>& that)
        : Base(that) {}

    //
    // Export cast (no code generated, reinterpret pointer).
    //

    template <typename OtherType>
    sysinline operator const GpuMatrix<OtherType>& () const
    {
        MATRIX__CHECK_CONVERSION(GpuPtr(Type), GpuPtr(OtherType));
        return recastEqualLayout<const GpuMatrix<OtherType>>(*this);
    }

};

//----------------------------------------------------------------

// Check the same size for both CPU and GPU
COMPILE_ASSERT(sizeof(GpuMatrix<uint8>) == (HEXLIB_GPU_BITNESS == 32 ? 16 : 24)); 
COMPILE_ASSERT(alignof(GpuMatrix<uint8>) == HEXLIB_GPU_BITNESS / 8);

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
sysinline const GpuMatrix<const Type>& makeConst(const GpuMatrix<Type>& matrix)
{
    return recastEqualLayout<const GpuMatrix<const Type>>(matrix);
}

//================================================================
//
// recastToNonConst
//
// Removes const qualifier from elements.
// Avoid using it!
//
//================================================================

template <typename Type>
sysinline const GpuMatrix<Type>& recastToNonConst(const GpuMatrix<const Type>& matrix)
{
    return recastEqualLayout<const GpuMatrix<Type>>(matrix);
}

//================================================================
//
// equalSize support
//
//================================================================

template <typename Type>
GET_SIZE_DEFINE(GpuMatrix<Type>, value.size())
