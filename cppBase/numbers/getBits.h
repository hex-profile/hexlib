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
    auto minValue = ldexp(Float(1), -maxBits);

    if (def(err) && err < minValue)
        err = minValue;

    return -nativeLog2(err);
}
