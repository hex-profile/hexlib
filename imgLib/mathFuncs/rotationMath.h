#pragma once

#include <cmath>

#include "point/point.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// pi32
// pi64
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
sysinline Point<Float> circleCCW(const Float& v);

//----------------------------------------------------------------

#if defined(__CUDA_ARCH__)

    template <>
    sysinline Point<float32> circleCCW(const float32& v)
    {
        float32 angle = v * (2 * pi32);
        Point<float32> result;
        __sincosf(angle, &result.Y, &result.X);
        return result;
    }

template <>
    sysinline Point<float64> circleCCW(const float64& v)
    {
        float64 angle = v * (2 * pi64);
        Point<float64> result;
        __sincos(angle, &result.Y, &result.X);
        return result;
    }

#elif defined(_GNU_SOURCE)

    template <>
    sysinline Point<float32> circleCCW(const float32& v)
    {
        float32 angle = v * (2 * pi32);
        Point<float32> result;
        sincosf(angle, &result.Y, &result.X);
        return result;
    }

    template <>
    sysinline Point<float64> circleCCW(const float64& v)
    {
        float64 angle = v * (2 * pi64);
        Point<float64> result;
        sincos(angle, &result.Y, &result.X);
        return result;
    }

#else

    template <>
    sysinline Point<float32> circleCCW(const float32& v)
    {
        float32 angle = v * (2 * pi32);
        return point(cosf(angle), sinf(angle));
    }

    template <>
    sysinline Point<float64> circleCCW(const float64& v)
    {
        float64 angle = v * (2 * pi64);
        return point(cos(angle), sin(angle));
    }

#endif

//================================================================
//
// complexMul
//
//================================================================

template <typename Float>
sysinline Float complexMulX(const Float& AX, const Float& AY, const Float& BX, const Float& BY)
    {return AX * BX - AY * BY;}

template <typename Float>
sysinline Float complexMulY(const Float& AX, const Float& AY, const Float& BX, const Float& BY)
    {return AX * BY + AY * BX;}

template <typename Float>
sysinline Point<Float> complexMul(const Point<Float>& A, const Point<Float>& B)
    {return point(A.X * B.X - A.Y * B.Y, A.X * B.Y + A.Y * B.X);}

//================================================================
//
// conjugate
//
//================================================================

template <typename Float>
sysinline Point<Float> conjugate(const Point<Float>& P)
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
sysinline Float getPhase(const Point<Float>& vec);

template <>
sysinline float32 getPhase(const Point<float32>& vec)
{
    float32 result = atan2f(vec.Y, vec.X);
    result *= (1 / (2 * pi32));
    if_not (def(result)) result = 0;
    return result;
}

template <>
sysinline float64 getPhase(const Point<float64>& vec)
{
    float64 result = atan2(vec.Y, vec.X);
    result *= (1 / (2 * pi64));
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

sysinline float32 approxPhase(const Point<float32>& value)
{
    float32 aX = absv(value.X);
    float32 aY = absv(value.Y);

    float32 minXY = minv(aX, aY);
    float32 maxXY = maxv(aX, aY);

    float32 D = nativeDivide(minXY, maxXY);
    if (maxXY == 0) D = 0; // range [0..1]

    // Cubic polynom approximation, at interval ends x=0 and x=1 both value and 1st derivative are equal to real function.
    float32 result = (0.1591549430918954f + ((-0.02288735772973838f) + (-0.01126758536215698f) * D) * D) * D;

    if (aY >= aX)
        result = 0.25f - result;

    if (value.X < 0)
        result = 0.5f - result;

    if (value.Y < 0)
        result = -result;

    return result;
}
