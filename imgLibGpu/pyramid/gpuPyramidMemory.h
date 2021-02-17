#pragma once

#include "dataAlloc/arrayObjectMemoryStatic.h"
#include "dataAlloc/gpuLayeredMatrixMemory.h"
#include "gpuAppliedApi/gpuAppliedApi.h"
#include "pyramid/gpuPyramid.h"
#include "pyramid/pyramidScale.h"

//================================================================
//
// gpuPyramidMemoryMaxLevels
//
//================================================================

static constexpr Space gpuPyramidMemoryMaxLevels = 32;

//================================================================
//
// GpuPyramidMemory<Type>
//
//================================================================

template <typename Type>
class GpuPyramidMemory : public GpuPyramid<Type>
{

public:

    Space levels() const
        {return pyramidArray.size();}

    Space layers() const
        {return currentLayers;}

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
            kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc, stdPassThru);
    }

public:

    template <typename Kit>
    inline stdbool realloc(const PyramidScale& scale, Space newLevels, Space newLayers, const Point<Space>& newBaseSize, Rounding sizeRounding, stdPars(Kit))
    {
        return reallocEx(scale, point(1.f), point(0), newLevels, newLayers, newBaseSize, sizeRounding, point(0), stdPassThru);
    }

private:

    using ImageStorage = GpuLayeredMatrixMemory<Type>;
    ArrayObjectMemoryStatic<ImageStorage, gpuPyramidMemoryMaxLevels> pyramidArray;
    Space currentLayers = 0;

    GpuLayeredMatrixNull<Type> emptyLayeredMatrix;

};
