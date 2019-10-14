#pragma once

#include "point/point.h"
#include "data/gpuPtr.h"

//================================================================
//
// GpuPyramidLevelLayout
//
// Untyped structure, with base address offset instead of base pointer.
//
//================================================================

struct GpuPyramidLevelLayout
{
    GpuAddrU memOffset;
    Point<Space> size;
    Space pitch;
    Space layerBytePitch;
};

//================================================================
//
// operator == (GpuPyramidLevelLayout, GpuPyramidLevelLayout)
//
//================================================================

sysinline bool operator ==(const GpuPyramidLevelLayout& a, const GpuPyramidLevelLayout& b)
{
    return
        a.memOffset == b.memOffset &&
        allv(a.size == b.size) &&
        a.pitch == b.pitch &&
        a.layerBytePitch == b.layerBytePitch;
}

//================================================================
//
// GpuPyramidLayout
//
// The structure is used to store the structure in GPU DRAM.
//
//================================================================

struct GpuPyramidLayout
{
    static constexpr Space maxLevels = 24;

    Space levels;
    Space layers;

    GpuPyramidLevelLayout levelData[maxLevels];
};

//================================================================
//
// initEmpty
//
//================================================================

sysinline void initEmpty(GpuPyramidLayout& layout)
{
    layout.levels = 0;
    layout.layers = 0;
}

//================================================================
//
// isEqualLayout
//
//================================================================

sysinline bool isEqualLayout(const GpuPyramidLayout& a, const GpuPyramidLayout& b)
{
    ensure(a.levels == b.levels);
    ensure(a.layers == b.layers);

    Space levels = a.levels;

    for (Space i = 0; i < levels; ++i)
        ensure(a.levelData[i] == b.levelData[i]);

    return true;
}

//================================================================
//
// GpuPyramidParam
//
// Structure for passing as a kernel parameter.
//
//================================================================

template <typename Type>
struct GpuPyramidParam
{
    GpuPtr(Type) basePointer;
    GpuPtr(GpuPyramidLayout) gpuLayout;
    CpuPtr(GpuPyramidLayout) cpuLayout;
    Space levels;
    Space layers;
};

//----------------------------------------------------------------

template <typename Type>
sysinline const GpuPyramidParam<const Type>& makeConst(const GpuPyramidParam<Type>& param)
{
    return recastEqualLayout<const GpuPyramidParam<const Type>>(param);
}
