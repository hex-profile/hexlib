#pragma once

#include "point/point.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// pi
// 
// (cannot use template constants because of MSVC bugs)
//
//================================================================

#ifndef ROTATION_PI_DEFINED
#define ROTATION_PI_DEFINED

constexpr float32 pi32 = 3.14159265358979324f;
constexpr float64 pi64 = 3.14159265358979324;

#endif

//================================================================
//
// circleCCW
//
// Counter-clockwise
//
//================================================================

template <typename Float>
sysinline Point<Float> circleCCW(Float v)
{
    Float angle = v * (2 * Float(pi64));
    Point<Float> result;
    nativeCosSin(angle, result.X, result.Y);
    return result;
}

//================================================================
//
// complexMul
//
//================================================================

template <typename Float>
sysinline Point<Float> complexMul(const Point<Float>& A, const Point<Float>& B)
    {return point(A.X * B.X - A.Y * B.Y, A.X * B.Y + A.Y * B.X);}

//================================================================
//
// complexFma
//
// Sometimes gives more efficient code on GPU.
//
//================================================================

sysinline Point<float32> complexFma(const Point<float32>& A, const Point<float32>& B, const Point<float32>& add)
{
    auto result = add;

    result.X += A.X * B.X;
    result.X -= A.Y * B.Y;
    result.Y += A.X * B.Y;
    result.Y += A.Y * B.X;

    return result;
}

//================================================================
//
// complexConjugate
//
//================================================================

template <typename Float>
sysinline Point<Float> complexConjugate(const Point<Float>& P)
    {return point(+P.X, -P.Y);}

//================================================================
//
// scalarProd
//
//================================================================

template <typename Float>
sysinline Float scalarProd(const Point<Float>& A, const Point<Float>& B)
    {return A.X * B.X + A.Y * B.Y;}

//================================================================
//
// getPhase
//
// Returns value in range [-1/2, +1/2]
//
//================================================================

template <typename Float>
sysinline Float getPhase(const Point<Float>& vec)
{
    Float result = nativeAtan2(vec.Y, vec.X);
    result *= (1 / (2 * Float(pi64)));
    if_not (def(result)) result = 0;
    return result;
}

//================================================================
//
// approxPhase
//
// The result is in range [-1/2, +1/2]
// On unit circle, max error is 0.00135.
//
//================================================================

template <typename Float>
sysinline Float approxPhase(const Point<Float>& value)
{
    Float aX = absv(value.X);
    Float aY = absv(value.Y);

    Float minXY = minv(aX, aY);
    Float maxXY = maxv(aX, aY);

    Float D = nativeDivide(minXY, maxXY);
    if (maxXY == 0) D = 0; // range [0..1]

    // Cubic polynom approximation, at interval ends x=0 and x=1 both value and 1st derivative are equal to real function.
    Float result = (0.1591549430918954f + ((-0.02288735772973838f) + (-0.01126758536215698f) * D) * D) * D;

    if (aY >= aX)
        result = 0.25f - result;

    if (value.X < 0)
        result = 0.5f - result;

    if (value.Y < 0)
        result = -result;

    return result;
}

//================================================================
//
// circularDistance
//
//================================================================

template <typename Float>
sysinline Float circularDistance(Float A, Float B) // A, B in [0..1) range 
{
    Float distance = A - B + 1; // [0, 2)
  
    if (distance >= 1) 
        distance -= 1; // [0, 1)

    if (distance >= 0.5f) 
        distance = 1 - distance; // [0, 1/2)

    return distance;
}
