#pragma once

#ifndef HEXLIB_MOVEMENT_3D
#define HEXLIB_MOVEMENT_3D

#include "extLib/types/pointTypes.h"

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
    // Quaternion of unit length.
    Point4D<Float> rotation{0, 0, 0, 1}; 

    // Translation.
    Point3D<Float> translation{0, 0, 0};
};

//----------------------------------------------------------------

template <typename Float>
HEXLIB_INLINE Movement3D<Float> movement3D(const Point4D<Float>& rotation, const Point3D<Float>& translation)
{
    Movement3D<Float> result;
    result.rotation = rotation;
    result.translation = translation;
    return result;
}

//----------------------------------------------------------------

#endif // #ifndef HEXLIB_MOVEMENT_3D
