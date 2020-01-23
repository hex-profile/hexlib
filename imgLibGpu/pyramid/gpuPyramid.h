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
    virtual Space levels() const =0;
    virtual Space layers() const =0;
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
inline bool equalSizePyramid(const GpuPyramid<T1>& p1, const GpuPyramid<T2>& p2)
{
    ensure(p1.levels() == p2.levels());

    Space levels = p1.levels();

    for (Space k = 0; k < levels; ++k)
        ensure(equalSize(p1.levelSize(k), p2.levelSize(k)));

    return true;
}

//----------------------------------------------------------------

template <typename T1, typename T2>
inline bool equalSize(const GpuPyramid<T1>& p1, const GpuPyramid<T2>& p2)
{
    return equalSizePyramid(p1, p2);
}

//----------------------------------------------------------------

template <typename T0, typename... Types>
sysinline bool equalSize(const GpuPyramid<T0>& v0, const GpuPyramid<Types>&... values)
{
    bool ok = true;
    char tmp[] = {(ok &= equalSizePyramid(v0, values), 'x')...};
    return ok;
}

//================================================================
//
// getLayers
//
//================================================================

template <typename Type>
sysinline Space getLayers(const GpuPyramid<Type>& pyramid)
{
    return pyramid.layers();
}
