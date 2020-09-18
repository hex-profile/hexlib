#pragma once

#include "extLib/types/compileTools.h"

//================================================================
//
// Point4D
//
//================================================================

template <typename Type>
struct Point4D
{
    Type X;
    Type Y;
    Type Z;
    Type W;
};

//================================================================
//
// point4D
//
// point function is used to create Point4D objects.
//
//================================================================

template <typename Type>
HEXLIB_INLINE Point4D<Type> point4D(const Type& X, const Type& Y, const Type& Z, const Type& W)
{
    Point4D<Type> result;
    result.X = X;
    result.Y = Y;
    result.Z = Z;
    result.W = W;
    return result;
}

template <typename Type>
HEXLIB_INLINE Point4D<Type> point4D(const Type& value)
{
    Point4D<Type> result;
    result.X = value;
    result.Y = value;
    result.Z = value;
    result.W = value;
    return result;
}
