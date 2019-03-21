#pragma once

#include "data/matrix.h"
#include "data/gpuPtr.h"
#include "data/gpuArray.h"

//================================================================
//
// GpuMatrix
//
// Matrix for GPU address space: identical to MatrixEx< GpuPtr(Type) >.
//
//================================================================

template <typename Type>
class GpuMatrix : public MatrixEx< GpuPtr(Type) >
{

public:

    using Base = MatrixEx< GpuPtr(Type) >;
    UseType_(Base, TmpType);

    //
    // Constructors
    //

    sysinline GpuMatrix(const TmpType* = 0)
        {}

    sysinline GpuMatrix(const ConstructUnitialized& tag)
        : Base(tag) {}

    sysinline GpuMatrix(GpuPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY)
        : Base(memPtr, memPitch, sizeX, sizeY) {}

    sysinline GpuMatrix(GpuPtr(Type) memPtr, Space memPitch, Space sizeX, Space sizeY, const MatrixPreconditions& preconditions)
        : Base(memPtr, memPitch, sizeX, sizeY, preconditions) {}

#ifdef DBGPTR_MODE

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
        COMPILE_ASSERT(sizeof(GpuMatrix<Type>) == sizeof(GpuMatrix<OtherType>));
        return * (const GpuMatrix<OtherType> *) this;
    }

};

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
inline const GpuMatrix<const Type>& makeConst(const GpuMatrix<Type>& matrix)
{
    COMPILE_ASSERT(sizeof(GpuMatrix<const Type>) == sizeof(GpuMatrix<Type>));
    return * (const GpuMatrix<const Type>*) &matrix;
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
inline const GpuMatrix<Type>& recastToNonConst(const GpuMatrix<const Type>& matrix)
{
    COMPILE_ASSERT(sizeof(GpuMatrix<const Type>) == sizeof(GpuMatrix<Type>));
    return * (const GpuMatrix<Type>*) &matrix;
}

//================================================================
//
// equalSize support
//
//================================================================

template <typename Type>
GET_SIZE_DEFINE(GpuMatrix<Type>, value.size())
