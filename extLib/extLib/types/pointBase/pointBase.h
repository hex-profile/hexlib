#pragma once

#ifndef HEXLIB_POINT_BASE
#define HEXLIB_POINT_BASE

#include "extLib/types/compileTools.h"

//================================================================
//
// Point
//
// Simple pair of X/Y values.
//
//================================================================

template <typename Type>
struct Point
{
    Type X;
    Type Y;
};

//================================================================
//
// point
//
// point function is used to create Point objects:
//
// point(1.f)
// point(2, 3)
//
//================================================================

template <typename Type>
HEXLIB_INLINE Point<Type> point(const Type& X, const Type& Y)
{
    Point<Type> result;
    result.X = X;
    result.Y = Y;
    return result;
}

template <typename Type>
HEXLIB_INLINE Point<Type> point(const Type& value)
{
    Point<Type> result;
    result.X = value;
    result.Y = value;
    return result;
}

//----------------------------------------------------------------

#endif // HEXLIB_POINT_BASE
