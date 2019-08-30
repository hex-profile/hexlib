#pragma once

#include "data/gpuMatrix.h"
#include "vectorTypes/vectorType.h"

//================================================================
//
// ChromaPackedType
//
//================================================================

template <typename Type>
struct ChromaPackedType
{
    using TypeSigned = TYPE_MAKE_SIGNED(Type);
    using T = typename MakeVectorType<TypeSigned, 2>::T;
};

//----------------------------------------------------------------

template <typename Type>
struct ChromaPackedType<const Type>
{
    using T = const typename ChromaPackedType<Type>::T;
};

//================================================================
//
// GpuPackedYuv
//
// YUV image in 4:2:0 format.
//
// Chroma is stored in packed format: 2-component vector elements.
//
//================================================================

template <typename LumaType>
struct GpuPackedYuv
{
    using ChromaType = typename ChromaPackedType<LumaType>::T;

    GpuMatrix<LumaType> luma;
    GpuMatrix<ChromaType> chroma;

    sysinline operator GpuPackedYuv<const LumaType> () const
        {return GpuPackedYuv<const LumaType>{luma, chroma};}
};

//================================================================
//
// equalSize
//
//================================================================

template <typename LumaType>
sysinline bool equalSize(const GpuPackedYuv<LumaType>& a, const GpuPackedYuv<LumaType>& b)
{
    return
        equalSize(a.luma, b.luma) &&
        equalSize(a.chroma, b.chroma);
}

//================================================================
//
// makeConst
//
//================================================================

template <typename Type>
sysinline const GpuPackedYuv<const Type>& makeConst(const GpuPackedYuv<Type>& image)
{
    return recastEqualLayout<const GpuPackedYuv<const Type>>(image);
}

//================================================================
//
// GpuPlanarYuv
//
// YUV image in 4:2:0 format.
//
// Chroma is stored in planar format.
//
//================================================================

template <typename LumaType>
struct GpuPlanarYuv
{
    using ChromaType = TYPE_MAKE_SIGNED(LumaType);

    GpuMatrix<LumaType> Y;
    GpuMatrix<ChromaType> U;
    GpuMatrix<ChromaType> V;

    sysinline GpuPlanarYuv() =default;

    sysinline operator GpuPlanarYuv<const LumaType> () const
        {return GpuPlanarYuv<const LumaType>{Y, U, V};}
};

//================================================================
//
// equalSize
//
//================================================================

template <typename LumaType>
sysinline bool equalSize(const GpuPlanarYuv<LumaType>& a, const GpuPlanarYuv<LumaType>& b)
{
    return
        equalSize(a.Y, b.Y) &&
        equalSize(a.U, b.U) &&
        equalSize(a.V, b.V);
}

//================================================================
//
// makeConst
//
//================================================================

template <typename Type>
sysinline const GpuPlanarYuv<const Type>& makeConst(const GpuPlanarYuv<Type>& image)
{
    return recastEqualLayout<const GpuPlanarYuv<const Type>>(image);
}
