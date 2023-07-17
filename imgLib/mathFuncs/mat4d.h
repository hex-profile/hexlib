#pragma once

#include "point4d/point4d.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// Mat4D
//
//================================================================

template <typename Float>
using Mat4D = Point4D<Point4D<Float>>;

//================================================================
//
// MAT4D_FOREACH
//
//================================================================

#define MAT4D_FOREACH(action) \
    \
    MAT4D_FOREACH_OUTER(X, action) \
    MAT4D_FOREACH_OUTER(Y, action) \
    MAT4D_FOREACH_OUTER(Z, action) \
    MAT4D_FOREACH_OUTER(W, action)

#define MAT4D_FOREACH_OUTER(p, action) \
    \
    action(p, X) \
    action(p, Y) \
    action(p, Z) \
    action(p, W)

//================================================================
//
// zeroMat4D()
//
//================================================================

template <typename Float>
sysinline auto zeroMat4D()
{
    Mat4D<Float> result;

    #define TMP_MACRO(a, b) \
        result.a.b = 0;

    MAT4D_FOREACH(TMP_MACRO)

    #undef TMP_MACRO

    return result;
}

//================================================================
//
// unitMat4D
//
//================================================================

template <typename Float>
sysinline auto unitMat4D()
{
    auto result = zeroMat4D<Float>();

    result.X.X = 1;
    result.Y.Y = 1;
    result.Z.Z = 1;
    result.W.W = 1;

    return result;
}

//================================================================
//
// Mat4D apply.
//
// 16 MADs.
//
//================================================================

template <typename Float>
sysinline Point4D<Float> operator %(const Mat4D<Float>& mat, const Point4D<Float>& vec)
{
    return point4D
    (
        scalarProd(mat.X, vec),
        scalarProd(mat.Y, vec),
        scalarProd(mat.Z, vec),
        scalarProd(mat.W, vec)
    );
}

//================================================================
//
// transpose
//
//================================================================

template <typename Float>
sysinline Mat4D<Float> transpose(const Mat4D<Float>& value)
{
    Mat4D<Float> result;

    #define TMP_MACRO(a, b) \
        result.a.b = value.b.a;

    MAT4D_FOREACH(TMP_MACRO)

    #undef TMP_MACRO

    return result;
}

//================================================================
//
// Mat4D combine.
//
// 64 MADs.
//
//================================================================

template <typename Float>
sysinline Mat4D<Float> operator %(const Mat4D<Float>& A, const Mat4D<Float>& B)
{
    auto Bt = transpose(B);

    Mat4D<Float> Rt;

    Rt.X = A % Bt.X;
    Rt.Y = A % Bt.Y;
    Rt.Z = A % Bt.Z;
    Rt.W = A % Bt.W;

    return transpose(Rt);
}
