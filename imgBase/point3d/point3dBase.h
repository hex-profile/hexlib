#pragma once

#ifndef HEXLIB_POINT3D_BASE
#define HEXLIB_POINT3D_BASE

//================================================================
//
// Point3D
//
//================================================================

template <typename Type>
struct Point3D
{
    Type X;
    Type Y;
    Type Z;
};

//================================================================
//
// point3D
//
// point function is used to create Point3D objects.
//
//================================================================

template <typename Type>
inline Point3D<Type> point3D(const Type& X, const Type& Y, const Type& Z)
{
    Point3D<Type> result;
    result.X = X;
    result.Y = Y;
    result.Z = Z;
    return result;
}

template <typename Type>
inline Point3D<Type> point3D(const Type& value)
{
    Point3D<Type> result;
    result.X = value;
    result.Y = value;
    result.Z = value;
    return result;
}

//----------------------------------------------------------------

#endif // HEXLIB_POINT3D_BASE
