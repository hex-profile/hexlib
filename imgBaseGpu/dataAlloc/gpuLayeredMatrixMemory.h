#pragma once

#include "data/gpuLayeredMatrix.h"
#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"
#include "gpuAppliedApi/gpuAppliedApi.h"

//================================================================
//
// GpuLayeredMatrixMemory<Type>
//
//================================================================

template <typename Type>
class GpuLayeredMatrixMemory : public GpuLayeredMatrix<Type>
{

    using AddrU = GpuAddrU;
    using Pointer = GpuPtr(Type);

public:

    sysinline GpuLayeredMatrixMemory()
        {initZero();}

    sysinline ~GpuLayeredMatrixMemory()
        {dealloc();}

private:

    GpuLayeredMatrixMemory(const GpuLayeredMatrixMemory<Type>& that); // forbidden
    void operator =(const GpuLayeredMatrixMemory<Type>& that); // forbidden

public:

    sysinline GpuLayeredMatrix<Type>& operator () ()
        {return *this;}

    sysinline const GpuLayeredMatrix<Type>& operator () () const
        {return *this;}

public:

    stdbool reallocEx(Space layers, const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit));

public:

    Space layers() const {return currentLayers;}

public:

    sysinline Space getImagePitch() const
        {return currentImagePitch;}

    sysinline Point<Space> getImageSize() const
        {return currentImageSize;}

    sysinline Pointer getBaseImagePtr() const
        {return allocPtr;}

    sysinline Space getLayerPitch() const
        {return currentLayerPitch;}

public:

    sysinline GpuMatrix<Type> getLayerInline(Space r) const
    {
        GpuMatrix<Type> result;

        if (SpaceU(r) < SpaceU(currentLayers))
        {
            Pointer resultPtr = allocPtr + r * currentLayerPitch;

            result.assign
            (
                resultPtr,
                currentImagePitch,
                currentImageSize.X, currentImageSize.Y,
                MatrixValidityAssertion{}
            );
        }

        return result;
    }

    sysinline GpuMatrix<Type> getLayer(Space r) const
    {
        return getLayerInline(r);
    }

public:

    void dealloc();

    ////

    sysinline Point<Space> size() const {return currentImageSize;}

    sysinline Point<Space> maxSize() const {return allocSize;}
    sysinline Space maxLayers() const {return allocLayers;}

    ////

    sysinline void resizeNull()
    {
        currentImageSize = point(0);
        currentImagePitch = 0;
        currentLayers = 0;
        currentLayerPitch = 0;
    }

    bool resize(Space layers, Space sizeX, Space sizeY); // rearrange without reallocation

    sysinline bool resize(Space layers, const Point<Space>& size)
        {return resize(layers, size.X, size.Y);}

private:

    sysinline void initZero()
    {
        currentImageSize = point(0);
        currentImagePitch = 0;
        currentLayers = 0;
        currentLayerPitch = 0;

        allocPtr = Pointer(0);
        allocSize = point(0);
        allocAlignMask = 0;
        allocLayers = 0;
        allocLayerPitch = 0;
    }

public:

    //
    // Default realloc: assumes kit.gpuFastAlloc
    //

    template <typename Kit>
    sysinline stdbool realloc(Space layers, const Point<Space>& size, stdPars(Kit))
        {return reallocEx(layers, size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc, stdPassThru);}

public:

    sysinline void releaseOwnership() {memoryOwner.discardAlloc(); initZero();}

private:

    MemoryOwner memoryOwner;

    //
    // Config params
    //

    Point<Space> currentImageSize;
    Space currentImagePitch;
    Space currentLayers;
    Space currentLayerPitch;

    //
    // Alloc params
    //

    Pointer allocPtr;
    Point<Space> allocSize;
    Space allocAlignMask;

    Space allocLayers;
    Space allocLayerPitch;

};

//================================================================
//
// GPU_LAYERED_MATRIX_ALLOC
//
//================================================================

#define GPU_LAYERED_MATRIX_ALLOC(name, Type, layers, size) \
    GpuLayeredMatrixMemory<Type> name; \
    require(name.realloc(layers, size, stdPass))
