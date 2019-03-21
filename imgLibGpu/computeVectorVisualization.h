#pragma once

#include "vectorTypes/vectorOperations.h"
#include "numbers/mathIntrinsics.h"

namespace computeVecVisualization {

//================================================================
//
// circularDistance
//
//================================================================

sysinline float32 circularDistance(float32 A, float32 B) // A, B in [0..1) range
{
    float32 distance = A - B + 1; // [0, 2)

    if (distance >= 1)
        distance -= 1; // [0, 1)

    if (distance >= 0.5f)
        distance = 1 - distance; // [0, 1/2)

    return distance;
}

//================================================================
//
// colorBspline
//
//================================================================

sysinline float32 colorBspline(float32 t)
{
    t = absv(t);

    float32 t2 = square(t);

    float32 result = -21.33333333f * t2 + 1;

    if (t >= 0.125f)
        result = 10.66666667f * t2 - 8*t + 1.5f;

    if (t >= 0.375f)
        result = 0;

    return result;
}

//================================================================
//
// computeVectorVisualization
//
//================================================================

template <bool useSaturation>
sysinline float32_x4 computeVectorVisualization(const float32_x2& value)
{

    //
    // vector to H(S)V
    //

    float32 H = 0;

    {
        float32 aX = absv(value.x);
        float32 aY = absv(value.y);

        float32 minXY = minv(aX, aY);
        float32 maxXY = maxv(aX, aY);

        float32 D = nativeDivide(minXY, maxXY);
        if (maxXY == 0) D = 0; /* range [0..1] */

        // Cubic polynom approximation, at interval ends x=0 and x=1 both value and 1st derivative are equal to real function.
        float32 result = (0.1591549430918954f + ((-0.02288735772973838f) + (-0.01126758536215698f) * D) * D) * D;

        if (aY >= aX)
            result = 0.25f - result;

        if (value.x < 0)
            result = 0.5f - result;

        if (value.y < 0)
            result = 1 - result;

        H = result; // range [0..1]
    }

    ////

    float32 length2 = square(value.x) + square(value.y);
    float32 length = fastSqrt(length2);

    //
    // Color conversion
    //

    #define HV_GET_CHANNEL(C, center) \
        \
        float32 C = colorBspline(circularDistance(H, center)); /* [0, 1] */ \

    HV_GET_CHANNEL(weightR, 0.f/4)
    HV_GET_CHANNEL(weightB, 1.f/4)
    HV_GET_CHANNEL(weightG, 2.f/4)
    HV_GET_CHANNEL(weightY, 3.f/4)

    ////

    float32_x4 colorR = make_float32_x4(0, 0, 1, 0);
    float32_x4 colorB = make_float32_x4(1, 0, 0, 0);
    float32_x4 colorG = make_float32_x4(0, 1, 0, 0);
    float32_x4 colorY = make_float32_x4(0, 0.5f, 0.5f, 0);

    float32_x4 pureColor =
        weightR * colorR +
        weightB * colorB +
        weightG * colorG +
        weightY * colorY;

    ////

    float32 saturation = saturate(length);

    float32_x4 resultColor = pureColor;

    if_not (useSaturation)
        resultColor = pureColor * length;
    else
    {
        float32 avgBrightness = (1.f/3) * (pureColor.x + pureColor.y + pureColor.z);
        float32_x4 equivalentGray = make_float32_x4(avgBrightness, avgBrightness, avgBrightness, 0);

        resultColor = linerp(saturation, equivalentGray, pureColor);
    }

    ////

    if_not (def(value.x) && def(value.y))
        resultColor = make_float32_x4(1, 0, 1, 0); // magenta

    ////

    return resultColor;
}

//================================================================
//
// limitColorBrightness
//
//================================================================

sysinline float32_x4 limitColorBrightness(const float32_x4& color)
{
    float32 maxComponent = maxv(color.x, color.y, color.z);
    float32_x4 result = color;

    if (maxComponent > 1.f)
    {
        float32 divMaxComponent = nativeRecip(maxComponent);
        result *= divMaxComponent;
    }

    return result;
}

//----------------------------------------------------------------

}

using computeVecVisualization::computeVectorVisualization;
using computeVecVisualization::limitColorBrightness;
