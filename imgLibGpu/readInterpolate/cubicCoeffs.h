#pragma once

#include "numbers/float/floatBase.h"

//================================================================
//
// cubicCoeffs
//
//================================================================

template <typename DstFloat>
sysinline void cubicCoeffs(float32 s, DstFloat& c0, DstFloat& c1, DstFloat& c2, DstFloat& c3)
{
    float32 s2 = s * s;
    float32 s3 = s2 * s;

    c0 = s2 + (-0.5f) * s3 + (-0.5f) * s;
    c1 = 1 + 1.5f * s3 + (-2.5f) * s2;
    c2 = (-1.5f) * s3 + 2 * s2 + 0.5f * s;
    c3 = 0.5f * s3 + (-0.5f) * s2;
}

//================================================================
//
// cubicBsplineCoeffs
//
// 11 instructions (mads)
//
//================================================================

template <typename DstFloat>
sysinline void cubicBsplineCoeffs(float32 s, DstFloat& c0, DstFloat& c1, DstFloat& c2, DstFloat& c3)
{
    float32 s2 = s * s;
    float32 s3 = s2 * s;

    float32 tmp = 0.1666666667f * s3;
    float32 aux = s2 - 0.5f * s3;

    c0 = 0.1666666667f - tmp + 0.5f * s2 - 0.5f * s;
    c1 = 0.6666666667f - aux;
    c2 = aux - 0.5f * s2 + 0.5f * s + 0.1666666667f;
    c3 = tmp;
}
