#pragma once

#include "data/gpuMatrix.h"
#include "data/gpuLayeredMatrix.h"
#include "gpuPyramidLayout.h"

//================================================================
//
// PyramidStructure
//
//================================================================

struct PyramidStructure
{
    virtual Space levelCount() const =0;
    virtual Space layerCount() const =0;
    virtual Point<Space> levelSize(Space level) const =0;
};

//================================================================
//
// GpuPyramidLayoutGetting
//
//================================================================

template <typename Type>
struct GpuPyramidLayoutGetting
{
    virtual bool getGpuLayout(GpuPtr(Type)& basePointer, GpuPyramidLayout& layout) const =0;
};

//================================================================
//
// GpuPyramid<T>
//
// Image pyramid interface.
//
//================================================================

template <typename Type>
struct GpuPyramid : public PyramidStructure, GpuPyramidLayoutGetting<Type>
{

    //----------------------------------------------------------------
    //
    // Main interface.
    //
    //----------------------------------------------------------------

    virtual GpuMatrix<Type> operator[] (Space level) const =0;
    virtual GpuMatrix<Type> getLayer(Space level, Space layer) const =0;

    virtual const GpuLayeredMatrix<Type>& getLevel(Space level) const =0;

    //----------------------------------------------------------------
    //
    // Cast from GpuPyramid<T> to GpuPyramid<const T>
    //
    //----------------------------------------------------------------

    inline operator GpuPyramid<const Type>& ()
        {return recastEqualLayout<GpuPyramid<const Type>>(*this);}

    inline operator const GpuPyramid<const Type>& () const
        {return recastEqualLayout<const GpuPyramid<const Type>>(*this);}

};

//================================================================
//
// equalSize
//
//================================================================

template <typename T1, typename T2>
inline bool equalSize(const GpuPyramid<T1>& p1, const GpuPyramid<T2>& p2)
{
    require(p1.levelCount() == p2.levelCount());

    Space levelCount = p1.levelCount();

    for (Space k = 0; k < levelCount; ++k)
        require(equalSize(p1.levelSize(k), p2.levelSize(k)));

    return true;
}

//================================================================
//
// getLayerCount
//
//================================================================

template <typename Type>
sysinline Space getLayerCount(const GpuPyramid<Type>& pyramid)
{
    return pyramid.layerCount();
}
