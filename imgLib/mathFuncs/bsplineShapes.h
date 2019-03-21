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

    static sysinline float32 coverageRadius()
        {return 0.5f;}

    static sysinline float32 meansqRadius()
        {return 0.2886751347f;}

    static sysinline float32 zeroValue()
        {return 1.f;}
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

    static sysinline float32 coverageRadius()
        {return 1.f;}

    static sysinline float32 meansqRadius()
        {return 0.4082482906f;}

    static sysinline float32 zeroValue()
        {return 1.f;}
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

    static sysinline float32 coverageRadius()
        {return 1.5f;}

    static sysinline float32 meansqRadius()
        {return 0.5f;}

    static sysinline float32 zeroValue()
        {return 0.75f;}

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

    static sysinline float32 coverageRadius()
        {return 2.f;}

    static sysinline float32 meansqRadius()
        {return 0.5773502693f;}

    static sysinline float32 zeroValue()
        {return 0.6666666667f;}

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

    static sysinline float32 coverageRadius()
        {return 2.5f;}

    static sysinline float32 meansqRadius()
        {return 0.6454972245f;}

    static sysinline float32 zeroValue()
        {return 0.5989583333f;}

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

    static sysinline float32 coverageRadius()
        {return 3.0f;}

    static sysinline float32 meansqRadius()
        {return 0.7071067810f;}

    static sysinline float32 zeroValue()
        {return 0.5500000000f;}

};

//================================================================
//
// Bspline<7>
//
//================================================================

template <>
struct Bspline<7>
{
    static sysinline float32 func(float32 x)
    {
        float32 t = absv(x);

        ////

        float32 result = 0.4793650794f + (-0.3333333333f + (0.1111111111f + (-0.2777777778e-1f + 0.6944444444e-2f * t) * t * t) * t * t) * t * t;

        if (t >= 1)
            result = 0.4904761905f + (-0.7777777778e-1f + (-0.1000000000f + (-0.3888888889f + (0.5f + (-0.2333333333f + (0.5000000000e-1f - 0.4166666667e-2f * t) * t) * t) * t) * t) * t) * t;

        if (t >= 2)
            result = -0.2206349206f + (2.411111111f + (-3.833333333f + (2.722222222f + (-1.055555556f + (.2333333333f + (-0.2777777778e-1f + 0.1388888889e-2f * t) * t) * t) * t) * t) * t) * t;

        if (t >= 3)
            result = 3.250793651f + (-5.688888889f + (4.266666667f + (-1.777777778f + (0.4444444444f + (-0.6666666667e-1f + (0.5555555556e-2f - 0.1984126984e-3f * t) * t) * t) * t) * t) * t) * t;

        if (t >= 4)
            result = 0;

        ////

        return result;
    }

    static sysinline float32 coverageRadius()
        {return 4.0f;}

    static sysinline float32 meansqRadius()
        {return 0.8164965809f;}

    static sysinline float32 zeroValue()
        {return 0.4793650794f;}

};
