#pragma once

#ifndef HEXLIB_MOVEMENT_3D
#define HEXLIB_MOVEMENT_3D

#include "types/pointTypes.h"

//================================================================
//
// Movement3D
//
// Application order: rotation, then translation.
//
//================================================================

template <typename Float>
struct Movement3D
{
    Point4D<Float> rotation; // quaternion of unit length
    Point3D<Float> translation;
};

//----------------------------------------------------------------

template <typename Float>
inline Movement3D<Float> movement3D(const Point4D<Float>& rotation, const Point3D<Float>& translation)
{
    Movement3D<Float> result;
    result.rotation = rotation;
    result.translation = translation;
    return result;
}

//----------------------------------------------------------------

#endif // #ifndef HEXLIB_MOVEMENT_3D
