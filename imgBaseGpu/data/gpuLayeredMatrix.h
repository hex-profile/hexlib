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

    inline operator const GpuLayeredMatrix<const Element>& () const
    {
        COMPILE_ASSERT(sizeof(GpuLayeredMatrix<Element>) == sizeof(GpuLayeredMatrix<const Element>));
        return * (const GpuLayeredMatrix<const Element>*) this;
    }

};

//================================================================
//
// PASS_LAYERED_MATRIX
//
//================================================================

#define PASS_LAYERED_MATRIX(n, name) \
    PREP_ENUM(n, PASS_LAYERED_MATRIX_FUNC, name)

#define PASS_LAYERED_MATRIX_FUNC(r, name) \
    name.getLayer(r)

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
// makeConst (fast)
//
//================================================================

template <typename Type>
inline const GpuLayeredMatrix<const Type>& makeConst(const GpuLayeredMatrix<Type>& matrix)
{
    COMPILE_ASSERT(sizeof(GpuLayeredMatrix<const Type>) == sizeof(GpuLayeredMatrix<Type>));
    return * (const GpuLayeredMatrix<const Type>*) &matrix;
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

    inline GpuLayeredMatrixFromMatrix(const GpuMatrix<Element>& base)
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
inline GpuLayeredMatrixFromMatrix<Element> gpuLayeredMatrixFromMatrix(const GpuMatrix<Element>& base)
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
