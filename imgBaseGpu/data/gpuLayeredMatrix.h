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
// GPU_LAYERED_MATRIX_PASS
//
//================================================================

#define GPU_LAYERED_MATRIX_PASS(n, name) \
    PREP_ENUM(n, GPU_LAYERED_MATRIX_PASS_FUNC, name)

#define GPU_LAYERED_MATRIX_PASS_FUNC(r, name) \
    name.getLayer(r)

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

////

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
    {return GpuLayeredMatrixFromMatrix<Element>(base);}

//================================================================
//
// GpuLayeredMatrixSubrange
//
//================================================================

template <typename Element>
class GpuLayeredMatrixSubrange : public GpuLayeredMatrix<Element>
{

public:

    Point<Space> size() const
        {return base.size();}

    Space layerCount() const
        {return actualEnd - actualBegin;}

    GpuMatrix<Element> getLayer(Space r) const
        {return base.getLayer(clampRange(r, actualBegin, actualEnd));}

public:

    GpuLayeredMatrixSubrange(const GpuLayeredMatrix<Element>& base, Space layerBegin, Space layerCount)
        : base(base)
    {
        Space baseCount = base.layerCount();
        actualBegin = clampRange(layerBegin, 0, baseCount);

        Space availCount = baseCount - actualBegin;
        actualEnd = actualBegin + clampRange(layerCount, 0, availCount);
    }

private:

    const GpuLayeredMatrix<Element>& base;
    Space actualBegin;
    Space actualEnd;

};
