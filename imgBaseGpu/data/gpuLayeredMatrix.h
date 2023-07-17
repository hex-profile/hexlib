#pragma once

#include "data/gpuMatrix.h"
#include "prepTools/prepEnum.h"

//================================================================
//
// GpuLayeredMatrix
//
//================================================================

template <typename Type>
struct GpuLayeredMatrix
{
    //
    // Base interface
    //

    virtual Point<Space> size() const =0;

    virtual Space layers() const =0;

    virtual GpuMatrix<Type> getLayer(Space r) const =0;

    //
    // Cast from GpuLayeredMatrix<T> to GpuLayeredMatrix<const T>
    //

    sysinline operator const GpuLayeredMatrix<const Type>& () const
    {
        return recastEqualLayout<const GpuLayeredMatrix<const Type>>(*this);
    }

};

//================================================================
//
// GpuLayeredMatrixNull
//
//================================================================

template <typename Type>
class GpuLayeredMatrixNull : public GpuLayeredMatrix<Type>
{
    virtual Point<Space> size() const
        {return point(0);}

    virtual Space layers() const
        {return 0;}

    virtual GpuMatrix<Type> getLayer(Space r) const
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

template <typename Type>
sysinline Space getLayers(const GpuLayeredMatrix<Type>& matrix)
    {return matrix.layers();}

//================================================================
//
// GpuLayeredMatrixFromMatrix
//
//================================================================

template <typename Type>
class GpuLayeredMatrixFromMatrix : public GpuLayeredMatrix<Type>
{

public:

    sysinline GpuLayeredMatrixFromMatrix(const GpuMatrix<Type>& base)
        : base(base) {}

    Point<Space> size() const
        {return base.size();}

    Space layers() const
        {return 1;}

    GpuMatrix<Type> getLayer(Space r) const
        {return r == 0 ? base : 0;}

public:

    sysinline const GpuLayeredMatrix<Type>& operator () () const
        {return *this;}

    sysinline GpuLayeredMatrix<Type>& operator () ()
        {return *this;}

private:

    GpuMatrix<Type> base;

};

//----------------------------------------------------------------

template <typename Type>
sysinline GpuLayeredMatrixFromMatrix<Type> layeredMatrix(const GpuMatrix<Type>& base)
{
    return GpuLayeredMatrixFromMatrix<Type>(base);
}

//================================================================
//
// GpuLayeredMatrixSubrange
//
//================================================================

template <typename Type>
class GpuLayeredMatrixSubrange : public GpuLayeredMatrix<Type>
{

public:

    sysinline GpuLayeredMatrixSubrange(const GpuLayeredMatrix<Type>& base, Space layerOrg, Space layerEnd)
        : base(base), layerOrg(layerOrg), layerEnd(layerEnd)
    {
    }

    virtual Point<Space> size() const
    {
        return base.size();
    }

    virtual Space layers() const
    {
        return isValid() ?
            layerEnd - layerOrg :
            0;
    }

    virtual GpuMatrix<Type> getLayer(Space i) const
    {
        return isValid() ?
            base.getLayer(layerOrg + i) :
            0;
    }

public:

    sysinline bool isValid() const
    {
        bool c0 = (0 <= layerOrg);
        bool c1 = (layerOrg <= layerEnd);
        bool c2 = (layerEnd <= base.layers());
        return c0 && c1 && c2; // Work around GCC madness.
    }

private:

    const GpuLayeredMatrix<Type>& base;
    Space const layerOrg;
    Space const layerEnd;

};
