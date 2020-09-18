#pragma once

#include "data/gpuMatrix.h"
#include "gpuSupport/parallelLoad.h"

//================================================================
//
// GpuSramPyramidLevel
//
//================================================================

template <typename Type>
class GpuSramPyramidLevel
{

public:

    sysinline Space layers() const 
        {return theLayers;}

    sysinline Point<Space> size() const 
        {return layout.size;}

    sysinline Space pitch() const 
        {return layout.pitch;}

    sysinline Space layerBytePitch() const
        {return layout.layerBytePitch;}

    sysinline Type* getUnsafePtr() const
        {return (Type*) (basePointer + layout.memOffset);}

public:

    template <Space groupSize, typename LoadMode>
    sysinline void loadNoSync(Space groupMember, const GpuPyramidParam<Type>& srcParam, Space level);

public:

    sysinline GpuMatrix<Type> getImage(Space layer = 0) const;

private:

    GpuAddrU basePointer;
    GpuPyramidLevelLayout layout;
    Space theLayers;

};

//================================================================
//
// GpuSramPyramidLevel::loadNoSync
//
//================================================================

#if DEVCODE

template <typename Type>
template <Space groupSize, typename LoadMode>
sysinline void GpuSramPyramidLevel<Type>::loadNoSync(Space groupMember, const GpuPyramidParam<Type>& srcParam, Space level)
{
    GpuPyramidLayout& src = *srcParam.gpuLayout;

    ////

    bool mainMember = (groupMember == 0);

    if (mainMember) this->basePointer = GpuAddrU(srcParam.basePointer);
    if (mainMember) this->theLayers = srcParam.layers;

    ////

    const GpuPyramidLevelLayout& srcLayout = src.levelData[level];
    GpuPyramidLevelLayout& dstLayout = this->layout;

    parallelLoadTypeNoSync<groupSize, LoadMode, GpuPyramidLevelLayout>(groupMember, srcLayout, dstLayout);
}

#endif

//================================================================
//
// GpuSramPyramidLevel::getImage
//
//================================================================

#if DEVCODE

template <typename Type>
sysinline GpuMatrix<Type> GpuSramPyramidLevel<Type>::getImage(Space layer) const
{
    devDebugCheck(SpaceU(layer) < SpaceU(theLayers));

    return GpuMatrix<Type>
    (
        (Type*) (basePointer + layout.memOffset + layer * layout.layerBytePitch),
        layout.pitch,
        layout.size.X, layout.size.Y,
        MatrixValidityAssertion{}
    );
}

#endif
