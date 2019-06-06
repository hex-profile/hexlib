#pragma once

#include "point/point.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// pi
//
//================================================================

#ifndef ROTATION_PI_DEFINED
#define ROTATION_PI_DEFINED

template <typename Float>
constexpr Float pi = Float(3.14159265358979324);

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
    Float angle = v * Float(2 * pi<Float>);
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
    {return A.X*B.X + A.Y*B.Y;}

//================================================================
//
// vectorLengthSq
//
//================================================================

template <typename Float>
sysinline Float vectorLengthSq(const Point<Float>& vec)
    {return square(vec.X) + square(vec.Y);}

//================================================================
//
// vectorLength
//
//================================================================

template <typename Float>
sysinline Float vectorLength(const Point<Float>& vec)
{
    Float lenSq = vectorLengthSq(vec);
    Float result = fastSqrt(lenSq);
    return result;
}

//================================================================
//
// vectorDecompose
//
//================================================================

template <typename Float>
sysinline void vectorDecompose(const Point<Float>& vec, Float& vectorLengthSq, Float& vectorDivLen, Float& vectorLength, Point<Float>& vectorDir)
{
    vectorLengthSq = square(vec.X) + square(vec.Y);
    vectorDivLen = recipSqrt(vectorLengthSq);
    vectorLength = vectorLengthSq * vectorDivLen;
    vectorDir = vec * vectorDivLen;

    if (vectorLengthSq == 0)
    {
        vectorLength = 0;
        vectorDir.X = 1;
        vectorDir.Y = 0;
    }
}

//================================================================
//
// vectorNormalize
//
//================================================================

template <typename Float>
sysinline Point<Float> vectorNormalize(const Point<Float>& vec)
{
    VECTOR_DECOMPOSE(vec);
    return vecDir;
}

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
    result *= (1 / (2 * pi<Float>));
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
