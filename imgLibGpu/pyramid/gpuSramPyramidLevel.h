#pragma once

#include "data/gpuMatrix.h"
#include "gpuSupport/parallelLoop.h"

//================================================================
//
// parallelLoadStructureNoSync
//
// Loads a structure from DRAM to SRAM by GPU group.
//
//================================================================

template <Space groupSize, typename LoadMode, typename Type>
sysinline void parallelLoadStructureNoSync(Space groupMember, const Type& srcParam, Type& dstParam)
{
    using Atom = uint32;

    ////

    COMPILE_ASSERT(sizeof(Type) % sizeof(Atom) == 0);
    const Space atomCount = sizeof(Type) / sizeof(Atom);
    COMPILE_ASSERT(alignof(Type) % alignof(Atom) == 0);

    ////

    const Atom* srcPtr = (const Atom*) &srcParam;
    Atom* dstPtr = (Atom*) &dstParam;

    ////

    srcPtr += groupMember;
    dstPtr += groupMember;

    ////

    PARALLEL_LOOP_UNBASED
    (
        i,
        atomCount,
        groupMember,
        groupSize,
        dstPtr[i] = LoadMode::func(&srcPtr[i])
    );
}

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

    parallelLoadStructureNoSync<groupSize, LoadMode, GpuPyramidLevelLayout>(groupMember, srcLayout, dstLayout);
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
        matrixPreconditionsAreVerified()
    );
}

#endif
