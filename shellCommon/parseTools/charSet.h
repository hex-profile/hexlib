#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// isAnySpace
//
//================================================================

template <typename Char>
sysinline bool isAnySpace(Char c)
{
    return
        c == ' ' ||
        c == '\t' ||
        c == '\v' ||
        c == '\r' ||
        c == '\n' ||
        c == '\f';
}

//================================================================
//
// isSpaceTab
//
//================================================================

template <typename Char>
sysinline bool isSpaceTab(Char c)
{
    return
        c == ' ' ||
        c == '\t';
}

//================================================================
//
// isDigit
//
//================================================================

template <typename Char>
sysinline bool isDigit(Char c)
{
    return c >= '0' && c <= '9';
}

//================================================================
//
// isHexDigit
//
//================================================================

template <typename Char>
sysinline bool isHexDigit(Char c)
{
    return
        (c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f');
}

//================================================================
//
// readAccumHexDigit
//
//================================================================

template <typename Value, typename Result>
sysinline bool readAccumHexDigit(Value c, Result& result)
{
    Result digit{};

    if (c >= '0' && c <= '9')
        digit = c - '0';

    else if (c >= 'A' && c <= 'F')
        digit = c - 'A' + 10;

    else if (c >= 'a' && c <= 'f')
        digit = c - 'a' + 10;

    else
        return false;

    ////

    result = (result << 4) + digit;

    return true;
}

//================================================================
//
// isLowerLetter
// isUpperLetter
//
//================================================================

template <typename Char>
sysinline bool isLowerLetter(Char c)
    {return (c >= 'a' && c <= 'z');}

template <typename Char>
sysinline bool isUpperLetter(Char c)
    {return (c >= 'A' && c <= 'Z');}

//================================================================
//
// toLowerLetter
// toUpperLetter
//
//================================================================

template <typename Char>
sysinline Char toLowerLetter(Char c)
{
    if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';

    return c;
}

template <typename Char>
sysinline Char toUpperLetter(Char c)
{
    if (c >= 'a' && c <= 'z')
        c += 'A' - 'a';

    return c;
}

//================================================================
//
// isLatinLetter
//
//================================================================

template <typename Char>
sysinline bool isLatinLetter(Char c)
{
    return
        (c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z');
}

//================================================================
//
// isDirSeparator
//
//================================================================

template <typename Char>
sysinline bool isDirSeparator(Char c)
{
    return c == '\\' || c == '/';
}

//================================================================
//
// isIdent1st
// isIdentNext
//
//================================================================

template <typename Char>
sysinline bool isIdent1st(Char c)
{
    return c == '_' || isLatinLetter(c);
}

//----------------------------------------------------------------

template <typename Char>
sysinline bool isIdentNext(Char c)
{
    return
        isIdent1st(c) ||
        isDigit(c);
}
