#pragma once

#include "point3d/point3d.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// Mat3D
//
//================================================================

template <typename Float>
using Mat3D = Point3D<Point3D<Float>>;

//================================================================
//
// MAT3D_FOREACH
//
//================================================================

#define MAT3D_FOREACH(action) \
    \
    MAT3D_FOREACH_OUTER(X, action) \
    MAT3D_FOREACH_OUTER(Y, action) \
    MAT3D_FOREACH_OUTER(Z, action)

#define MAT3D_FOREACH_OUTER(p, action) \
    \
    action(p, X) \
    action(p, Y) \
    action(p, Z)

//================================================================
//
// zeroMat3D()
//
//================================================================

template <typename Float>
sysinline auto zeroMat3D()
{
    Mat3D<Float> result;

    #define TMP_MACRO(a, b) \
        result.a.b = 0;

    MAT3D_FOREACH(TMP_MACRO)

    #undef TMP_MACRO

    return result;
}

//================================================================
//
// unitMat3D
//
//================================================================

template <typename Float>
sysinline auto unitMat3D()
{
    auto result = zeroMat3D<Float>();

    result.X.X = 1;
    result.Y.Y = 1;
    result.Z.Z = 1;

    return result;
}

//================================================================
//
// Mat3D apply.
//
// 16 MADs.
//
//================================================================

template <typename Float>
sysinline Point3D<Float> operator %(const Mat3D<Float>& mat, const Point3D<Float>& vec)
{
    return point3D
    (
        scalarProd(mat.X, vec),
        scalarProd(mat.Y, vec),
        scalarProd(mat.Z, vec)
    );
}

//================================================================
//
// transpose
//
//================================================================

template <typename Float>
sysinline Mat3D<Float> transpose(const Mat3D<Float>& value)
{
    Mat3D<Float> result;

    #define TMP_MACRO(a, b) \
        result.a.b = value.b.a;

    MAT3D_FOREACH(TMP_MACRO)

    #undef TMP_MACRO

    return result;
}

//================================================================
//
// Mat3D combine.
//
// 64 MADs.
//
//================================================================

template <typename Float>
sysinline Mat3D<Float> operator %(const Mat3D<Float>& A, const Mat3D<Float>& B)
{
    auto Bt = transpose(B);

    Mat3D<Float> Rt;

    Rt.X = A % Bt.X;
    Rt.Y = A % Bt.Y;
    Rt.Z = A % Bt.Z;

    return transpose(Rt);
}
