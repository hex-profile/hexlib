#pragma once

#include "dataAlloc/arrayObjMemStatic.h"
#include "dataAlloc/gpuLayeredMatrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "pyramid/gpuPyramid.h"
#include "pyramid/pyramidScale.h"

//================================================================
//
// pyramidMemoryMaxLevels
//
//================================================================

static constexpr Space pyramidMemoryMaxLevels = 32;

//================================================================
//
// PyramidMemory<Type>
//
//================================================================

template <typename Type>
class PyramidMemory : public GpuPyramid<Type>
{

public:

    Space levelCount() const
        {return pyramidArray.size();}

    Space layerCount() const
        {return currentLayerCount;}

    Point<Space> levelSize(Space level) const;

public:

    GpuMatrix<Type> operator[] (Space level) const;
    GpuMatrix<Type> getLayer(Space level, Space layer) const;
    const GpuLayeredMatrix<Type>& getLevel(Space level) const;

public:

    bool getGpuLayout(GpuPtr(Type)& basePointer, GpuPyramidLayout& layout) const;

public:

    inline GpuPyramid<Type>& operator () () {return *this;}

public:

    stdbool reallocEx
    (
        const PyramidScale& scale,
        const Point<float32>& baseScaleFactor,
        const Point<Space>& baseScaleLevels,
        Space newLevels,
        Space newLayers,
        const Point<Space>& newBaseSize,
        Rounding sizeRounding,
        const Point<Space>& extraEdge,
        Space baseByteAlignment,
        Space rowByteAlignment,
        AllocatorObject<GpuAddrU>& allocator,
        stdPars(ErrorLogKit)
    );

    void dealloc();

public:

    template <typename Kit>
    inline stdbool reallocEx
    (
        const PyramidScale& scale,
        const Point<float32>& baseScaleFactor,
        const Point<Space>& baseScaleLevels,
        Space newLevels,
        Space newLayers,
        const Point<Space>& newBaseSize,
        Rounding sizeRounding,
        const Point<Space>& extraEdge,
        stdPars(Kit)
    )
    {
        return reallocEx(scale, baseScaleFactor, baseScaleLevels, newLevels, newLayers, newBaseSize, sizeRounding, extraEdge,
            kit.gpuProperties.samplerBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc, stdPassThru);
    }

public:

    template <typename Kit>
    inline stdbool realloc(const PyramidScale& scale, Space newLevels, Space newLayers, const Point<Space>& newBaseSize, Rounding sizeRounding, stdPars(Kit))
    {
        return reallocEx(scale, point(1.f), point(0), newLevels, newLayers, newBaseSize, sizeRounding, point(0), stdPassThru);
    }

private:

    using ImageStorage = GpuLayeredMatrixMemory<Type>;
    ArrayObjMemStatic<ImageStorage, pyramidMemoryMaxLevels> pyramidArray;
    Space currentLayerCount = 0;

    GpuLayeredMatrixEmpty<Type> emptyLayeredMatrix;

};
