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
//----------------------------------------------------------------
//
// Allocation parameters are documented at PyramidConfigOptions class.
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
        const Point<Space>& newBaseSize,
        Space newLevels,
        Space newLayers,
        const PyramidScale& scale,
        Rounding sizeRounding,
        const PyramidConfigOptions& options,
        Space baseByteAlignment,
        Space rowByteAlignment,
        AllocatorInterface<GpuAddrU>& allocator,
        stdPars(ErrorLogKit)
    );

    void dealloc();

public:

    template <typename Kit>
    inline stdbool realloc
    (
        const Point<Space>& newBaseSize,
        Space newLevels,
        Space newLayers,
        const PyramidScale& scale,
        Rounding sizeRounding,
        const PyramidConfigOptions& options,
        stdPars(Kit)
    )
    {
        return reallocEx
        (
            newBaseSize, newLevels, newLayers, scale, sizeRounding, options,
            kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, kit.gpuFastAlloc,
            stdPassThru
        );
    }

public:

    template <typename Kit>
    inline stdbool realloc
    (
        const Point<Space>& newBaseSize,
        Space newLevels,
        Space newLayers,
        const PyramidScale& scale,
        Rounding sizeRounding, 
        stdPars(Kit)
    )
    {
        return realloc(newBaseSize, newLevels, newLayers, scale, sizeRounding, PyramidConfigOptions{}, stdPassThru);
    }

private:

    using ImageStorage = GpuLayeredMatrixMemory<Type>;
    ArrayObjectMemoryStatic<ImageStorage, gpuPyramidMemoryMaxLevels> pyramidArray;
    Space currentLayers = 0;

    GpuLayeredMatrixNull<Type> emptyLayeredMatrix;

};
