#pragma once

#include "vectorTypes/vectorOperations.h"
#include "numbers/mathIntrinsics.h"
#include "mathFuncs/rotationMath.h"
#include "mathFuncs/gaussApprox.h"

namespace computeVecVisualization {

//================================================================
//
// colorSupport
//
//================================================================

sysinline float32 colorSupport(float32 t)
{
    // Gausses are joined on sigma = 0.55f
    constexpr float32 sigma = 0.55f;
    constexpr float32 divSigma = 1 / sigma;
    return gaussExpoApprox<4>(square(t * divSigma));
}

//================================================================
//
// computeVectorVisualization
//
//================================================================

sysinline float32_x4 computeVectorVisualization(const float32_x2& value, bool grayMode = false)
{

    //
    // vector to H(S)V
    //

    float32 H = approxPhase(point(value.x, value.y)); 
    if (H < 0) H += 1;

    float32 length = vectorLength(value);

    //
    // Color conversion
    //

    #define TMP_MACRO(C, center) \
        float32 C = colorSupport(4.f * circularDistance(H, center)); /* [0, 1] */ \

    TMP_MACRO(weightR, 0.f/4)
    TMP_MACRO(weightB, 1.f/4)
    TMP_MACRO(weightG, 2.f/4)
    TMP_MACRO(weightY, 3.f/4)

    #undef TMP_MACRO

    ////

    auto colorR = make_float32_x4(0, 0, 1, 0);
    auto colorB = make_float32_x4(1, 0, 0, 0);
    auto colorG = make_float32_x4(0, 1, 0, 0);
    auto colorY = make_float32_x4(0, 1.f, 1.f, 0);

    float32 divSumWeight = fastRecip(weightR + weightB + weightG + weightY);
    auto pureColor = divSumWeight * (weightR * colorR + weightB * colorB + weightG * colorG + weightY * colorY);

    ////

    if (grayMode)
        pureColor = make_float32_x4(1, 1, 1, 0);

    ////

    auto resultColor = pureColor * length;

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
    auto result = color;

    if (maxComponent > 1.f)
    {
        float32 divMaxComponent = fastRecip(maxComponent);
        result *= divMaxComponent;
    }

    return result;
}

//----------------------------------------------------------------

}

using computeVecVisualization::computeVectorVisualization;
using computeVecVisualization::limitColorBrightness;
