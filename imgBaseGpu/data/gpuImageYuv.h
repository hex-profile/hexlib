#pragma once

#include "data/gpuMatrix.h"
#include "vectorTypes/vectorType.h"

//================================================================
//
// MakeChromaType
//
//================================================================

template <typename Type>
struct MakeChromaType
{
    using TypeSigned = TYPE_MAKE_SIGNED(Type);
    using T = typename MakeVectorType<TypeSigned, 2>::T;
};

//----------------------------------------------------------------

template <typename Type>
struct MakeChromaType<const Type>
{
    using T = const typename MakeChromaType<Type>::T;
};

//================================================================
//
// GpuImageYuv
//
// YUV image in 4:2:0 format.
//
// Chroma is stored in packed format: 2-component vector elements.
//
//================================================================

template <typename LumaType>
struct GpuImageYuv
{
    using ChromaType = typename MakeChromaType<LumaType>::T;

    GpuMatrix<LumaType> luma;
    GpuMatrix<ChromaType> chroma;

    sysinline GpuImageYuv()
        {}

    sysinline GpuImageYuv(const GpuMatrix<LumaType>& luma, const GpuMatrix<ChromaType>& chroma)
        : luma(luma), chroma(chroma) {}

    sysinline operator GpuImageYuv<const LumaType> () const
        {return GpuImageYuv<const LumaType>(luma, chroma);}
};

//================================================================
//
// equalSize
//
//================================================================

template <typename LumaType>
sysinline bool equalSize(const GpuImageYuv<LumaType>& m1, const GpuImageYuv<LumaType>& m2)
{
    return
        equalSize(m1.luma, m2.luma) &&
        equalSize(m1.chroma, m2.chroma);
}

//================================================================
//
// makeConst
//
//================================================================

template <typename Type>
sysinline const GpuImageYuv<const Type>& makeConst(const GpuImageYuv<Type>& image)
{
    return recastEqualLayout<const GpuImageYuv<const Type>>(image);
}
