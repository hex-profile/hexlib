#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intType.h"

//================================================================
//
// readHexByte
//
//================================================================

template <typename Iterator, typename Result>
inline bool readHexByte(Iterator& ptr, Iterator end, Result& result)
{
    auto p = ptr;

    result = 0;
    ensure(p != end && readAccumHexDigit(*p++, result));
    ensure(p != end && readAccumHexDigit(*p++, result));

    ptr = p;
    return true;
}

//================================================================
//
// readUint
//
//================================================================

template <typename Iterator, typename Uint>
inline bool readUint(Iterator& ptr, Iterator end, Uint& result)
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Uint));

    auto s = ptr;

    Uint value = 0;
    constexpr Uint maxValue = TYPE_MAX(Uint);
    constexpr Uint maxValueDiv10 = maxValue / 10;

    for (; s != end && isDigit(*s); ++s)
    {
        ensure(value <= maxValueDiv10);
        value *= 10;

        Uint digit = *s - '0';
        ensure(value <= maxValue - digit);
        value += digit;
    }

    ensure(s != ptr);

    ptr = s;
    result = value;
    return true;
}

//================================================================
//
// readInt
//
// Doesn't support INT_MIN.
//
//================================================================

template <typename Iterator, typename Int>
inline bool readInt(Iterator& ptr, Iterator end, Int& result)
{
    COMPILE_ASSERT(TYPE_IS_BUILTIN_INT(Int));

    auto s = ptr;

    ////

    bool invert = false;

    if (s != end && (*s == '-' || *s == '+'))
    {
        invert = (*s == '-');
        ++s;
    }

    ////

    Int value = 0;
    ensure(readUint(s, end, value));

    ////

    if (invert)
    {
        value = -value;
        ensure(value < 0);
    }

    ////

    ptr = s;
    result = value;
    return true;
}

//================================================================
//
// readFloatApprox
//
//================================================================

template <typename Iterator, typename Float>
inline bool readFloatApprox(Iterator& ptr, Iterator end, Float& result)
{
    Iterator s = ptr;

    //
    // Sign.
    //

    bool invert = false;

    if (s != end && (*s == '-' || *s == '+'))
    {
        invert = (*s == '-');
        ++s;
    }

    //
    // Int part.
    //

    auto bodyStart = s;

    Float value = 0;

    for (; s != end && isDigit(*s); ++s)
    {
        int digit = (*s - '0');
        value = value * 10 + digit;
    }

    //
    // Frac part.
    //

    if (s != end && *s == '.')
    {
        ++s;

        Float factor = Float(0.1);

        for (; s != end && isDigit(*s); ++s)
        {
            int digit = (*s - '0');
            value = value + factor * digit;
            factor *= Float(0.1);
        }
    }

    ensure(s != bodyStart);

    //
    // Exponent.
    //

    if (s != end && (*s == 'e' || *s == 'E'))
    {
        ++s;

        int exponent = 0;
        ensure(readInt(s, end, exponent));

        ////

        constexpr int maxExpo = 12;

        static const Float expoArray[] =
        {
            Float(1e-12),
            Float(1e-11),
            Float(1e-10),
            Float(1e-09),
            Float(1e-08),
            Float(1e-07),
            Float(1e-06),
            Float(1e-05),
            Float(1e-04),
            Float(1e-03),
            Float(1e-02),
            Float(1e-01),

            Float(1),

            Float(1e+01),
            Float(1e+02),
            Float(1e+03),
            Float(1e+04),
            Float(1e+05),
            Float(1e+06),
            Float(1e+07),
            Float(1e+08),
            Float(1e+09),
            Float(1e+10),
            Float(1e+11),
            Float(1e+12)
        };

        COMPILE_ASSERT(COMPILE_ARRAY_SIZE(expoArray) == 2 * maxExpo + 1);
        auto expos = expoArray + maxExpo;

        ////

        if (absv(exponent) >= 8 * maxExpo)
        {
            constexpr Float C = Float(2.302585092994045684017991454684364);
            value *= exp(C * exponent);
        }
        else
        {
            while (!(exponent >= -maxExpo))
            {
                value *= expos[-maxExpo];
                exponent += maxExpo;
            }

            while (!(exponent <= +maxExpo))
            {
                value *= expos[+maxExpo];
                exponent -= maxExpo;
            }

            ensure(exponent >= -maxExpo && exponent <= +maxExpo);

            value *= expos[exponent];
        }
    }

    //
    // Invert.
    //

    if (invert)
        value = -value;

    ////

    ptr = s;
    result = value;
    return true;
}
