#pragma once

#include "numbers/mathIntrinsics.h"

//================================================================
//
// getBits
//
//================================================================

template <typename Float>
sysinline Float getBits(Float err, int maxBits = 24)
{
    auto minValue = ldexpv(Float(1), -maxBits);

    if (def(err) && err < minValue)
        err = minValue;

    return -fastLog2(err);
}

//================================================================
//
// getBits
//
//================================================================

template <typename Float>
sysinline Point<Float> getBits(const Point<Float>& err, int maxBits = 24)
{
    return point(getBits(err.X, maxBits), getBits(err.Y, maxBits));
}
