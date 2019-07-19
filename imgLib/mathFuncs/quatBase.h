#pragma once

#include "point3d/point3dBase.h"
#include "point4d/point4dBase.h"

//================================================================
//
// quatReal
// quatImaginary
// quatCompose
//
//================================================================

template <typename Float>
sysinline Float quatReal(const Point4D<Float>& Q)
{
    return Q.W;
}

template <typename Float>
sysinline Point3D<Float> quatImaginary(const Point4D<Float>& Q)
{
    return {Q.X, Q.Y, Q.Z};
}

template <typename Float>
sysinline Point4D<Float> quatCompose(Float real, const Point3D<Float>& vec)
{
    return {vec.X, vec.Y, vec.Z, real};
}

//================================================================
//
// quatIdentity
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatIdentity()
{
    return quatCompose<Float>(1, point3D<Float>(0));
}
