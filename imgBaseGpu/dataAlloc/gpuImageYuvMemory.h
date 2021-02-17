#pragma once

#include "dataAlloc/gpuMatrixMemory.h"
#include "data/gpuImageYuv.h"
#include "numbers/divRound.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// GpuPackedYuvMemory
//
//================================================================

template <typename LumaType>
class GpuPackedYuvMemory
{

public:

    using ChromaType = typename ChromaPackedType<LumaType>::T;

    GpuMatrixMemory<LumaType> luma;
    GpuMatrixMemory<ChromaType> chroma;

public:

    sysinline operator GpuPackedYuv<LumaType> () const
        {return GpuPackedYuv<LumaType>{luma, chroma};}

    sysinline operator GpuPackedYuv<const LumaType> () const
        {return GpuPackedYuv<const LumaType>{luma, chroma};}

    sysinline GpuPackedYuv<LumaType> operator()()
        {return GpuPackedYuv<LumaType>{luma, chroma};}

public:

    sysinline GpuPackedYuvMemory() {}

public:

    sysinline void dealloc()
    {
        luma.dealloc();
        chroma.dealloc();
    }

    template <typename Kit>
    inline stdbool realloc
    (
        const Point<Space>& size,
        Space baseByteAlignment,
        Space rowByteAlignment,
        Rounding sizeRounding,
        stdPars(Kit)
    )
    {
        Point<Space> chromaSize = divNonneg(size, point(2), sizeRounding);

        REMEMBER_CLEANUP_EX(deallocCleanup, dealloc());

        require(luma.reallocEx(size, baseByteAlignment, rowByteAlignment, kit.gpuFastAlloc, stdPassThru));
        require(chroma.reallocEx(chromaSize, baseByteAlignment, rowByteAlignment, kit.gpuFastAlloc, stdPassThru));

        deallocCleanup.cancel();
        returnTrue;
    }

    template <typename Kit>
    sysinline stdbool realloc(const Point<Space>& size, Rounding sizeRounding, stdPars(Kit))
    {
        return realloc(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.gpuProperties.samplerRowAlignment, sizeRounding, stdPassThru);
    }

};
