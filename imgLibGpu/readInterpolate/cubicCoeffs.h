#pragma once

#include "numbers/float/floatBase.h"

//================================================================
//
// CubicCoeffs
//
//================================================================

struct CubicCoeffs
{
    template <typename Float, typename DstFloat>
    static sysinline void func(Float s, DstFloat& c0, DstFloat& c1, DstFloat& c2, DstFloat& c3)
    {
        Float s2 = s * s;
        Float s3 = s2 * s;

        c0 = DstFloat(s2 + (-0.5f) * s3 + (-0.5f) * s);
        c1 = DstFloat(1 + 1.5f * s3 + (-2.5f) * s2);
        c2 = DstFloat((-1.5f) * s3 + 2 * s2 + 0.5f * s);
        c3 = DstFloat(0.5f * s3 + (-0.5f) * s2);
    }
};

//================================================================
//
// CubicBsplineCoeffs
//
// 11 instructions (mads)
//
//================================================================

struct CubicBsplineCoeffs
{
    template <typename Float, typename DstFloat>
    static sysinline void func(Float s, DstFloat& c0, DstFloat& c1, DstFloat& c2, DstFloat& c3)
    {
        Float s2 = s * s;
        Float s3 = s2 * s;

        Float tmp = 0.1666666667f * s3;
        Float aux = s2 - 0.5f * s3;

        c0 = DstFloat(0.1666666667f - tmp + 0.5f * s2 - 0.5f * s);
        c1 = DstFloat(0.6666666667f - aux);
        c2 = DstFloat(aux - 0.5f * s2 + 0.5f * s + 0.1666666667f);
        c3 = DstFloat(tmp);
    }
};
