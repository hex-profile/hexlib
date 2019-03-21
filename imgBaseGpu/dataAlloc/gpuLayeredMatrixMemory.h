#pragma once

#include "data/gpuLayeredMatrix.h"
#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"

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

    inline GpuLayeredMatrixMemory()
        {initZero();}

    inline ~GpuLayeredMatrixMemory()
        {dealloc();}

private:

    GpuLayeredMatrixMemory(const GpuLayeredMatrixMemory<Type>& that); // forbidden
    void operator =(const GpuLayeredMatrixMemory<Type>& that); // forbidden

public:

    bool reallocEx(Space layerCount, const Point<Space>& size, Space baseByteAlignment, Space rowByteAlignment, AllocatorObject<AddrU>& allocator, stdPars(ErrorLogKit));

public:

    Space layerCount() const {return currentLayerCount;}

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

        if (SpaceU(r) < SpaceU(currentLayerCount))
        {
            Pointer resultPtr = allocPtr + r * currentLayerPitch;

            result.assign
            (
                resultPtr,
                currentImagePitch,
                currentImageSize.X, currentImageSize.Y,
                matrixPreconditionsAreVerified()
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

    inline Point<Space> size() const {return currentImageSize;}

    inline Point<Space> maxSize() const {return allocSize;}
    inline Space maxLayerCount() const {return allocLayerCount;}

    ////

    inline void resizeNull()
    {
        currentImageSize = point(0);
        currentImagePitch = 0;
        currentLayerCount = 0;
        currentLayerPitch = 0;
    }

    bool resize(Space layerCount, Space sizeX, Space sizeY); // rearrange without reallocation

    inline bool resize(Space layerCount, const Point<Space>& size)
        {return resize(layerCount, size.X, size.Y);}

private:

    inline void initZero()
    {
        currentImageSize = point(0);
        currentImagePitch = 0;
        currentLayerCount = 0;
        currentLayerPitch = 0;

        allocPtr = Pointer(0);
        allocSize = point(0);
        allocAlignMask = 0;
        allocLayerCount = 0;
        allocLayerPitch = 0;
    }

public:

    //
    // Default realloc: assumes kit.gpuFastAlloc
    //

    template <typename Kit>
    inline bool realloc(Space layerCount, const Point<Space>& size, stdPars(Kit))
        {return reallocEx(layerCount, size, kit.gpuProperties.samplerBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc, stdPassThru);}

public:

    inline void releaseOwnership() {memoryOwner.discardAlloc(); initZero();}

private:

    //
    MemoryOwner memoryOwner;

    //
    // Config params
    //

    Point<Space> currentImageSize;
    Space currentImagePitch;
    Space currentLayerCount;
    Space currentLayerPitch;

    //
    // Alloc params
    //

    Pointer allocPtr;
    Point<Space> allocSize;
    Space allocAlignMask;

    Space allocLayerCount;
    Space allocLayerPitch;

};

//================================================================
//
// GPU_LAYERED_MATRIX_ALLOC
//
//================================================================

#define GPU_LAYERED_MATRIX_ALLOC(name, Type, layerCount, size) \
    GpuLayeredMatrixMemory<Type> name; \
    require(name.realloc(layerCount, size, stdPass))
