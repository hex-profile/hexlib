#pragma once

#include "dataAlloc/gpuMatrixMemory.h"
#include "data/gpuImageYuv.h"
#include "numbers/divRound.h"

//================================================================
//
// GpuImageYuvMemory
//
//================================================================

template <typename LumaType>
class GpuImageYuvMemory
{

public:

    using TypeSigned = TYPE_MAKE_SIGNED(LumaType);
    using ChromaType = typename MakeVectorType<TypeSigned, 2>::T;

    GpuMatrixMemory<LumaType> luma;
    GpuMatrixMemory<ChromaType> chroma;

public:

    sysinline operator GpuImageYuv<LumaType> () const
        {return GpuImageYuv<LumaType>(luma, chroma);}

    sysinline operator GpuImageYuv<const LumaType> () const
        {return GpuImageYuv<const LumaType>(luma, chroma);}

    sysinline GpuImageYuv<LumaType> operator()()
        {return GpuImageYuv<LumaType>(luma, chroma);}

public:

    sysinline GpuImageYuvMemory() {}

public:

    sysinline void dealloc()
    {
        luma.dealloc();
        chroma.dealloc();
    }

    template <typename Kit>
    sysinline bool realloc
    (
        const Point<Space>& size,
        Space baseByteAlignment,
        Space rowByteAlignment,
        Rounding sizeRounding,
        stdPars(Kit)
    )
    {
        Point<Space> chromaSize = divNonneg(size, point(2), sizeRounding);

        bool okLuma = luma.reallocEx(size, baseByteAlignment, rowByteAlignment, kit.gpuFastAlloc, stdPassThru);
        bool okChroma = chroma.reallocEx(chromaSize, baseByteAlignment, rowByteAlignment, kit.gpuFastAlloc, stdPassThru);

        if_not (okLuma && okChroma)
            {dealloc(); return false;}

        return true;
    }

    template <typename Kit>
    sysinline bool realloc(const Point<Space>& size, Rounding sizeRounding, stdPars(Kit))
    {
        return realloc(size, kit.gpuProperties.samplerBaseAlignment, kit.gpuProperties.samplerRowAlignment, sizeRounding, stdPassThru);
    }

};
