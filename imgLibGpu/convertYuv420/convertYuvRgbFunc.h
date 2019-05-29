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
// convertYPbPrToBgr (unclamped)
//
//================================================================

sysinline float32_x4 convertYPbPrToBgr(float32 Ys, float32 Pb, float32 Pr)
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
// convertBgrToYPbPr
//
//================================================================

sysinline void convertBgrToYPbPr(const float32_x4& bgrValue, float32& Ys, float32& Pb, float32& Pr)
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
// convertRgbToPCA
// 
// PCA colorspace is a projection to the first three PCA vectors of natural images.
//
//================================================================

sysinline float32_x4 convertBgrToPCA(const float32_x4& bgrValue)
{
    // BGRx maps to (x, y, z, w)
    float32 B = bgrValue.x;
    float32 G = bgrValue.y;
    float32 R = bgrValue.z;

    float32 c0 = +0.5773502693f * R + 0.5773502693f * G + 0.5773502693f * B; // Brightness
    float32 cX = -0.7071067810f * R + 0.7071067810f * B; // Red-Blue channel
    float32 cY = -0.4082482906f * R + 0.8164965809f * G - 0.4082482906f * B; // Green-Magenta channel

    return make_float32_x4(c0, cX, cY, 0);
}

//================================================================
//
// convertPCAToBgr
//
//================================================================

sysinline float32_x4 convertPCAToBgr(const float32_x4& opponentValue)
{
    float32 c0 = opponentValue.x;
    float32 cX = opponentValue.y;
    float32 cY = opponentValue.z;

    float32 R = +0.5773502693f * c0 - 0.7071067811f * cX - 0.4082482906f * cY;
    float32 G = +0.5773502693f * c0 + 0.8164965809f * cY; 
    float32 B = +0.5773502693f * c0 + 0.7071067816f * cX - 0.4082482906f * cY;

    return make_float32_x4(B, G, R, 0);
}
