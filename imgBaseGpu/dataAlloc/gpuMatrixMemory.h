#pragma once

#include "data/gpuMatrix.h"
#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLogKit.h"
#include "gpuAppliedApi/gpuAppliedApi.h"

//================================================================
//
// GpuMatrixMemory<Type>
//
// Allocates/frees matrix memory.
// Does NOT call element constructors and destructors.
//
// Gpu-based address space.
//
//================================================================

template <typename Type>
class GpuMatrixMemory : public GpuMatrix<Type>
{

    using AddrU = GpuAddrU;
    using BaseMatrix = GpuMatrix<Type>;
    using Pointer = GpuPtr(Type);

public:

    sysinline GpuMatrixMemory()
        {initZero();}

    sysinline ~GpuMatrixMemory()
        {dealloc();}

private:

    GpuMatrixMemory(const GpuMatrixMemory<Type>& that); // forbidden
    void operator =(const GpuMatrixMemory<Type>& that); // forbidden

public:

    //
    // Export cast (no code generated, reinterpret 'this')
    //

    sysinline operator const GpuMatrix<Type>& () const
    {
        const GpuMatrix<Type>* baseMatrix = this;
        return *baseMatrix;
    }

    sysinline operator const GpuMatrix<const Type>& () const
    {
        const GpuMatrix<Type>* baseMatrix = this;
        return recastEqualLayout<const GpuMatrix<const Type>>(*baseMatrix);
    }

    sysinline const GpuMatrix<Type>& operator()() const
    {
        const GpuMatrix<Type>* baseMatrix = this;
        return *baseMatrix;
    }

public:

    stdbool reallocEx(const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit));

    ////

    void dealloc();

    ////

    sysinline bool allocated() const {return allocPtr != Pointer(0);}

    ////

    sysinline Space maxSizeX() const {return allocSize.X;}
    sysinline Space maxSizeY() const {return allocSize.Y;}
    sysinline Point<Space> maxSize() const {return allocSize;}

    ////

    sysinline void resizeNull()
        {BaseMatrix::assignNull();}

    bool resize(Space sizeX, Space sizeY); // rearrange without reallocation

    sysinline bool resize(const Point<Space>& size)
        {return resize(size.X, size.Y);}

private:

    sysinline void initZero()
    {
        allocPtr = Pointer(0);
        allocSize = point(0);
        allocAlignMask = 0;
    }

public:

    //
    // Default realloc: assumes kit.gpuFastAlloc
    //

    template <typename Kit>
    sysinline stdbool realloc(const Point<Space>& size, stdPars(Kit))
        {return reallocEx(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc, stdPassThru);}

    template <typename Kit>
    sysinline stdbool realloc(const Point<Space>& size, Space rowByteAlignment, stdPars(Kit))
        {return reallocEx(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, rowByteAlignment, kit.gpuFastAlloc, stdPassThru);}

public:

    sysinline void releaseOwnership() {memoryOwner.discardAlloc(); initZero();}

private:

    //
    MemoryOwner memoryOwner;

    // Only for resize support
    Pointer allocPtr;
    Point<Space> allocSize;
    Space allocAlignMask;

};

//================================================================
//
// GPU_MATRIX_ALLOC
//
//================================================================

#define GPU_MATRIX_ALLOC(name, Type, size) \
    GpuMatrixMemory<Type> name; \
    require(name.realloc(size, stdPass));
