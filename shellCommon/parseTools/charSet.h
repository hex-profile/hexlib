#pragma once

#include "charType/charType.h"

//================================================================
//
// isAnySpace
//
//================================================================

template <typename Char>
inline bool isAnySpace(Char c)
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
inline bool isSpaceTab(Char c)
{
    return
        c == ' ' ||
        c == '\t';
}

//================================================================
//
// isNewLine
//
//================================================================

template <typename Char>
inline bool isNewLine(Char c)
{
    return
        c == '\r' ||
        c == '\n';
}

//================================================================
//
// isDigit
//
//================================================================

template <typename Char>
inline bool isDigit(Char c)
{
    return c >= '0' && c <= '9';
}

//================================================================
//
// isBigHexDigit
//
//================================================================

template <typename Char>
inline bool isBigHexDigit(Char c)
{
    return
        (c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F');
}

//================================================================
//
// isLowerLetter
// isUpperLetter
//
//================================================================

template <typename Char>
inline bool isLowerLetter(Char c)
    {return (c >= 'a' && c <= 'z');}

template <typename Char>
inline bool isUpperLetter(Char c)
    {return (c >= 'A' && c <= 'Z');}

//================================================================
//
// toLowerLetter
// toUpperLetter
//
//================================================================

template <typename Char>
inline Char toLowerLetter(Char c)
{
    if (c >= 'A' && c <= 'Z')
        c += 'a' - 'A';

    return c;
}

template <typename Char>
inline Char toUpperLetter(Char c)
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
inline bool isLatinLetter(Char c)
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
inline bool isDirSeparator(Char c)
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
inline bool isIdent1st(Char c)
{
    return c == '_' || isLatinLetter(c);
}

//----------------------------------------------------------------

template <typename Char>
inline bool isIdentNext(Char c)
{
    return
        isIdent1st(c) ||
        isDigit(c);
}
