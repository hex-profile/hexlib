#pragma once

#include "numbers/float/floatType.h"

//================================================================
//
// Bspline
//
//================================================================

template <int32 order>
struct Bspline;

//================================================================
//
// Bspline<0> (box)
//
//================================================================

template <>
struct Bspline<0>
{
    static sysinline float32 func(float32 x)
    {
        float32 result = 0;

        if (absv(x) <= 0.5f)
            result = 1;

        return result;
    }

    static constexpr float32 coverageRadius = 0.5f;
    static constexpr float32 meansqRadius = 0.2886751347f;
    static constexpr float32 zeroValue = 1.f;
};

//================================================================
//
// Bspline<1> (tent)
//
//================================================================

template <>
struct Bspline<1>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        float32 result = 0;

        if (t <= 1)
            result = 1 - t;

        return result;
    }

    static constexpr float32 coverageRadius = 1.f;
    static constexpr float32 meansqRadius = 0.4082482906f;
    static constexpr float32 zeroValue = 1.f;
};

//================================================================
//
// Bspline<2>
//
//================================================================

template <>
struct Bspline<2>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        float32 result = -t*t + 0.75f;

        if (t >= 0.5f)
            result = 1.125f + (-1.5f + 0.5f * t) * t;

        if (t >= 1.5f)
            result = 0;

        return result;
    }

    static constexpr float32 coverageRadius = 1.5f;
    static constexpr float32 meansqRadius = 0.5f;
    static constexpr float32 zeroValue = 0.75f;
};

//================================================================
//
// Bspline<3>
//
//================================================================

template <>
struct Bspline<3>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        ////

        float32 result = 0.6666666667f + (-1 + 0.5f * t) * t * t;

        if (t >= 1)
            result = 1.333333333f + (-2 + (1 - 0.1666666667f * t) * t) * t;

        if (t >= 2)
            result = 0;

        ////

        return result;
    }

    static constexpr float32 coverageRadius = 2.f;
    static constexpr float32 meansqRadius = 0.5773502693f;
    static constexpr float32 zeroValue = 0.6666666667f;
};

//================================================================
//
// Bspline<4>
//
//================================================================

template <>
struct Bspline<4>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        ////

        float32 result = 0.5989583333f + (-0.625f + 0.25f * t * t) * t * t;

        if (t >= 0.5f)
            result = 0.5729166667f + (0.2083333333f + (-1.25f + (0.8333333333f - 0.1666666667f * t) * t) * t) * t;

        if (t >= 1.5f)
            result = 1.627604167f + (-2.604166667f + (1.5625f + (-0.4166666667f + 0.04166666667f * t) * t) * t) * t;

        if (t >= 2.5f)
            result = 0;

        ////

        return result;
    }

    static constexpr float32 coverageRadius = 2.5f;
    static constexpr float32 meansqRadius = 0.6454972245f;
    static constexpr float32 zeroValue = 0.5989583333f;
};

//================================================================
//
// Bspline<5>
//
//================================================================

template <>
struct Bspline<5>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        ////

        float32 result = 0.55f + (-0.5f + (0.25f - 0.8333333333e-1f * t) * t * t) * t * t;

        if (t >= 1)
            result = 0.425f + (0.625f + (-1.75f + (1.25f + (-0.375f + 0.4166666667e-1f * t) * t) * t) * t) * t;

        if (t >= 2)
            result = 2.025f + (-3.375f + (2.25f + (-0.75f + (0.125f - 0.8333333333e-2f * t) * t) * t) * t) * t;

        if (t >= 3)
            result = 0;

        ////

        return result;
    }

    static constexpr float32 coverageRadius = 3.0f;
    static constexpr float32 meanSqRadius = 0.7071067810f;
    static constexpr float32 zeroValue = 0.5500000000f;
};
