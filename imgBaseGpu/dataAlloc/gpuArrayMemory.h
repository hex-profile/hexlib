#pragma once

#include "dataAlloc/arrayMemory.h"
#include "data/gpuPtr.h"
#include "data/gpuArray.h"

//================================================================
//
// GpuArrayMemory<Type>
//
// Allocates/frees array memory.
// Does NOT call element constructors and destructors.
//
// GPU-based address space.
//
//================================================================

template <typename Type>
class GpuArrayMemory : public ArrayMemoryEx<GpuPtr(Type)>
{

    using Base = ArrayMemoryEx<GpuPtr(Type)>;
    using AddrU = typename Base::AddrU;

public:

    //
    // realloc
    //

    using Base::realloc;

    template <typename Kit>
    stdbool realloc(Space size, Space byteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit))
        {return Base::realloc(size, byteAlignment, allocator, stdPassThru);}

    template <typename Kit>
    inline stdbool realloc(Space size, Space byteAlignment, stdPars(Kit))
        {return Base::realloc(size, byteAlignment, kit.gpuFastAlloc, stdPassThru);}

    template <typename Kit>
    inline stdbool realloc(Space size, stdPars(Kit))
        {return Base::realloc(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuFastAlloc, stdPassThru);}

    //
    // Cast to GpuArray
    //

    inline operator const GpuArray<Type>& () const
    {
        const ArrayEx<GpuPtr(Type)>* arr = this;
        return recastEqualLayout<const GpuArray<Type>>(*arr);
    }

    inline operator const GpuArray<const Type>& () const
    {
        const ArrayEx<GpuPtr(Type)>* arr = this;
        return recastEqualLayout<const GpuArray<const Type>>(*arr);
    }

    inline const GpuArray<Type>& operator () () const
    {
        const ArrayEx<GpuPtr(Type)>* arr = this;
        return recastEqualLayout<const GpuArray<Type>>(*arr);
    }

};

//================================================================
//
// GPU_ARRAY_ALLOC
//
//================================================================

#define GPU_ARRAY_ALLOC(name, Type, size) \
    GpuArrayMemory<Type> name; \
    require(name.realloc(size, stdPass));

//================================================================
//
// matrix
//
//================================================================

template <typename Type>
sysinline auto matrix(const GpuArrayMemory<Type>& arr)
{
    return matrix(arr());
}
