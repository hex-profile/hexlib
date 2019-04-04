#pragma once

#include "vectorTypes/vectorBase.h"
#include "vectorTypes/vectorOperations.h"
#include "numbers/mathIntrinsics.h"

//================================================================
//
// Our YUV:
//
// Pb/Pr are doubled in range [-1, +1].
// Y is also in range [-1, +1].
//
//================================================================

//================================================================
//
// convertYPrPbToBgr (unclamped)
//
//================================================================

sysinline float32_x4 convertYPrPbToBgr(float32 Ys, float32 Pr, float32 Pb)
{
    // ITU-R BT.601, but our components are doubled and Y is in range [-1, +1]
    float32 Yu = Ys * 0.5f + 0.5f;

    float32 R = Yu + (0.5f * 1.402f) * Pr;
    float32 G = Yu + (0.5f * -0.7141362862f) * Pr + (0.5f * -0.3441362862f) * Pb;
    float32 B = Yu + (0.5f * 1.772f) * Pb;

    return make_float32_x4(B, G, R, 0);
}

//================================================================
//
// convertBgrToYPrPb
//
//================================================================

sysinline void convertBgrToYPrPb(const float32_x4& bgrValue, float32& Ys, float32& Pr, float32& Pb)
{
    // BGRx maps to (x, y, z, w)
    float32 R = bgrValue.z;
    float32 G = bgrValue.y;
    float32 B = bgrValue.x;

    // ITU-R BT.601, but our components are doubled and Y is in range [-1, +1]
    Ys = R * (2 * 0.299f) + G * (2 * 0.587f) + B * (2 * 0.114f) - 1; // [-1, +1]

    Pb = (2 * -0.16873589f) * R + (2 * -0.33126411f) * G + (2 * +0.500000000f) * B; // [-1, +1]
    Pr = (2 * +0.50000000f) * R + (2 * -0.41868759f) * G + (2 * -0.081312411f) * B; // [-1, +1]
}

//================================================================
//
// convertRgbToOpponent
//
//================================================================

sysinline float32_x4 convertBgrToOpponent(const float32_x4& bgrValue)
{
    // BGRx maps to (x, y, z, w)
    float32 R = bgrValue.z;
    float32 G = bgrValue.y;
    float32 B = bgrValue.x;

    float32 brightness = (1/3.f) * R + (1/3.f) * G + (1/3.f) * B;

    float32 colorX = R - G;
    float32 colorY = B + (-0.5f) * R + (-0.5f) * G;

    return make_float32_x4(colorX, colorY, brightness, 0);
}

//================================================================
//
// convertOpponentToBgr
//
//================================================================

sysinline float32_x4 convertOpponentToBgr(const float32_x4& opponentValue)
{
    float32 cX = opponentValue.x;
    float32 cY = opponentValue.y;
    float32 L = opponentValue.z;

    float32 tmp = L + (-1.f/3) * cY;

    float32 R = 0.5f * cX + tmp;
    float32 G = -0.5f * cX + tmp;
    float32 B = L + (2.f/3) * cY;

    return make_float32_x4(B, G, R, 0);
}
