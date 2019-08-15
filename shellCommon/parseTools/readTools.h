#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intBase.h"

//================================================================
//
// readHexDigit
//
//================================================================

template <typename Value>
inline int readHexDigit(Value c)
{
    int result = -1;

    if (c >= '0' && c <= '9')
        result = c - '0';

    if (c >= 'A' && c <= 'F')
        result = c - 'A' + 10;

    if (c >= 'a' && c <= 'f')
        result = c - 'a' + 10;

    return result;
}

//================================================================
//
// readHexByte
//
//================================================================

template <typename Iterator, typename Result>
inline bool readHexByte(Iterator& ptr, Iterator end, Result& result)
{
    Iterator s = ptr;

    ensure(s != end);
    auto v0 = readHexDigit(*s++);

    ensure(s != end);
    auto v1 = readHexDigit(*s++);

    ensure(v0 >= 0 && v1 >= 0);

    result = 16 * v0 + v1;

    ptr = s;
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
    Iterator s = ptr;

    Uint value = 0;
    constexpr Uint maxValue = TYPE_MAX(Uint);
    constexpr Uint maxValueDiv10Floor = maxValue / 10;

    for (; s != end && isDigit(*s); ++s)
    {
        ensure(value <= maxValueDiv10Floor);
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
// (doesn't support INT_MIN)
//
//================================================================

template <typename Iterator, typename Int>
inline bool readInt(Iterator& ptr, Iterator end, Int& result)
{
    Iterator s = ptr;

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

        constexpr Float C = Float(2.302585092994045684017991454684364);
        value *= exp(C * exponent);
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
