#pragma once

#include "point3d/point3dBase.h"
#include "point4d/point4dBase.h"
#include "compileTools/compileTools.h"

//================================================================
//
// quatConj
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatConj(const Point4D<Float>& Q)
{
    return {Q.X, -Q.Y, -Q.Z, -Q.W};
}

//================================================================
//
// quatMul
//
//================================================================

template <typename Float>
sysinline Point4D<Float> quatMul(const Point4D<Float>& A, const Point4D<Float>& B)
{
    return
    {
        B.X * A.X - B.Y * A.Y - B.Z * A.Z - B.W * A.W,
        B.X * A.Y + B.Y * A.X - B.Z * A.W + B.W * A.Z,
        B.X * A.Z + B.Y * A.W + B.Z * A.X - B.W * A.Y,
        B.X * A.W - B.Y * A.Z + B.Z * A.Y + B.W * A.X
    };
}

//================================================================
//
// quatRotateVec
//
// The quaternion should be normalized.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> quatRotateVec(const Point3D<Float>& V, const Point4D<Float>& Q)
{
    auto t0 = V.X * Q.Y + V.Y * Q.Z + V.Z * Q.W;
    auto t1 = V.X * Q.X - V.Y * Q.W + V.Z * Q.Z;
    auto t2 = V.X * Q.W + V.Y * Q.X - V.Z * Q.Y;
    auto t3 = V.X * Q.Z - V.Y * Q.Y - V.Z * Q.X;

    return
    {
        t0 * Q.Y + t1 * Q.X - t2 * Q.W - t3 * Q.Z,
        t0 * Q.Z + t1 * Q.W + t2 * Q.X + t3 * Q.Y,
        t0 * Q.W - t1 * Q.Z + t2 * Q.Y - Q.X * t3
    };
}
