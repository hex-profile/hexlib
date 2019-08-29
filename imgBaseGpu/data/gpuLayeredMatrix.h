#pragma once

#include "data/gpuMatrix.h"
#include "prepTools/prepEnum.h"

//================================================================
//
// GpuLayeredMatrix
//
//================================================================

template <typename Element>
struct GpuLayeredMatrix
{

    //
    // Base interface
    //

    virtual Point<Space> size() const =0;

    virtual Space layerCount() const =0;

    virtual GpuMatrix<Element> getLayer(Space r) const =0;

    //
    // Cast from GpuLayeredMatrix<T> to GpuLayeredMatrix<const T>
    //

    sysinline operator const GpuLayeredMatrix<const Element>& () const
    {
        return recastEqualLayout<const GpuLayeredMatrix<const Element>>(*this);
    }

};

//================================================================
//
// GpuLayeredMatrixEmpty
//
//================================================================

template <typename Element>
class GpuLayeredMatrixEmpty : public GpuLayeredMatrix<Element>
{
    Point<Space> size() const
        {return point(0);}

    Space layerCount() const
        {return 0;}

    GpuMatrix<Element> getLayer(Space r) const
        {return 0;}
};

//================================================================
//
// GPU_LAYERED_MATRIX_PASS
//
//================================================================

#define GPU_LAYERED_MATRIX_PASS(n, name) \
    PREP_ENUM(n, GPU_LAYERED_MATRIX_PASS_ITEM, name)

#define GPU_LAYERED_MATRIX_PASS_COMMA(n, name) \
    PREP_ENUMERATE(n, GPU_LAYERED_MATRIX_PASS_ITEM, name)

#define GPU_LAYERED_MATRIX_PASS_ITEM(r, name) \
    (name).getLayer(r)

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
sysinline const GpuLayeredMatrix<const Type>& makeConst(const GpuLayeredMatrix<Type>& matrix)
{
    return recastEqualLayout<const GpuLayeredMatrix<const Type>>(matrix);
}

//================================================================
//
// GetSize
//
//================================================================

template <typename Type>
GET_SIZE_DEFINE(GpuLayeredMatrix<Type>, value.size())

sysinline Space getLayerCount(Space layerCount)
    {return layerCount;}

template <typename Type>
sysinline Space getLayerCount(const GpuLayeredMatrix<Type>& matrix)
    {return matrix.layerCount();}

//================================================================
//
// GpuLayeredMatrixFromMatrix
//
//================================================================

template <typename Element>
class GpuLayeredMatrixFromMatrix : public GpuLayeredMatrix<Element>
{

public:

    sysinline GpuLayeredMatrixFromMatrix(const GpuMatrix<Element>& base)
        : base(base) {}

    Point<Space> size() const
        {return base.size();}

    Space layerCount() const
        {return 1;}

    GpuMatrix<Element> getLayer(Space r) const
        {return r == 0 ? base : 0;}

private:

    GpuMatrix<Element> base;

};

//----------------------------------------------------------------

template <typename Element>
sysinline GpuLayeredMatrixFromMatrix<Element> gpuLayeredMatrixFromMatrix(const GpuMatrix<Element>& base)
{
    return GpuLayeredMatrixFromMatrix<Element>(base);
}
