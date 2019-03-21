#pragma once

#include "numbers/mathIntrinsics.h"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// xyzPreShaper
//
//================================================================

sysinline float32 xyzPreShaper(float32 v)
{

    float32 result = 0.07739938080f * v;

    if (v > 0.04045f)
        result = powf(0.9478672986f * v + 0.05213270142f, 2.4f);

    return result;
}

//================================================================
//
// labPreShaper
//
//================================================================

sysinline float32 labPreShaper(float32 v)
{
    float32 result = 7.787f * v + 0.1379310345f;

    if (v > 0.008856f)
        result = powf(v, 1.f/3);

    return result;
}

//================================================================
//
// convertBgrToLab
//
//================================================================

sysinline float32_x4 convertBgrToLab(const float32_x4& bgr)
{
    // BGRx maps to (x, y, z, _)
    float32 sR = xyzPreShaper(bgr.z);
    float32 sG = xyzPreShaper(bgr.y);
    float32 sB = xyzPreShaper(bgr.x);

    float32 sX = labPreShaper(0.43389060160f * sR + 0.3762349154f * sG + 0.1899060465f * sB);
    float32 sY = labPreShaper(0.21260000000f * sR + 0.7152000000f * sG + 0.0722000000f * sB);
    float32 sZ = labPreShaper(0.01772544842f * sR + 0.1094753084f * sG + 0.8729553741f * sB);

    float32 L = 1.16f * sY - 0.16f;
    float32 a = 5.00f * sX - 5.00f * sY;
    float32 b = 2.00f * sY - 2.00f * sZ;

    return make_float32_x4(L, a, b, 0);
}
