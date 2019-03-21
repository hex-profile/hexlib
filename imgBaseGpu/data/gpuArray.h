#pragma once

#include "data/array.h"
#include "data/gpuPtr.h"

//================================================================
//
// GpuArray<Type>
//
// Array for GPU address space: identical to ArrayEx< GpuPtr(Type) >.
//
//================================================================

template <typename Type>
class GpuArray : public ArrayEx< GpuPtr(Type) >
{

public:

    using Base = ArrayEx< GpuPtr(Type) >;

    //
    // Construct
    //

    sysinline GpuArray()
        {}

    sysinline GpuArray(GpuPtr(Type) ptr, Space size)
        : Base(ptr, size) {}

    sysinline GpuArray(const Base& base)
        : Base(base) {}

    //
    // Export cast (no code generated, reinterpret 'this')
    //

    template <typename OtherType>
    sysinline operator const GpuArray<OtherType>& () const
    {
        ARRAY__CHECK_CONVERSION(GpuPtr(Type), GpuPtr(OtherType));
        COMPILE_ASSERT(sizeof(GpuArray<Type>) == sizeof(GpuArray<OtherType>));
        return * (const GpuArray<OtherType> *) this;
    }

};

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
inline const GpuArray<const Type>& makeConst(const GpuArray<Type>& array)
{
    COMPILE_ASSERT(sizeof(GpuArray<const Type>) == sizeof(GpuArray<Type>));
    return * (const GpuArray<const Type>*) &array;
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
inline const GpuArray<Type>& recastToNonConst(const GpuArray<const Type>& array)
{
    COMPILE_ASSERT(sizeof(GpuArray<const Type>) == sizeof(GpuArray<Type>));
    return * (const GpuArray<Type>*) &array;
}
