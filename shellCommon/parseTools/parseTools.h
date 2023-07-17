#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"
#include "numbers/int/intType.h"

//================================================================
//
// skipSpaceTab
//
//================================================================

template <typename Iterator>
sysinline bool skipSpaceTab(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    while (s != end && isSpaceTab(*s))
        ++s;

    bool advance = (s != ptr);
    ptr = s;

    return advance;
}

//================================================================
//
// skipAnySpace
//
//================================================================

template <typename Iterator>
sysinline bool skipAnySpace(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    while (s != end && isAnySpace(*s))
        ++s;

    bool advance = (s != ptr);
    ptr = s;

    return advance;
}

//================================================================
//
// skipNonSpaceCharacters
//
//================================================================

template <typename Iterator>
sysinline bool skipNonSpaceCharacters(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    while (s != end && !isSpaceTab(*s))
        ++s;

    bool advance = (s != ptr);
    ptr = s;
    return advance;
}

//================================================================
//
// skipIdent
//
// Parses a C/C++ identifier.
//
// Returns an indication of successful parsing.
// If parsing is not successful, does not move the pointer.
//
//================================================================

template <typename Iterator>
sysinline bool skipIdent(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    ensure(s != end && isIdent1st(*s));

    ++s;

    while (s != end && isIdentNext(*s))
        ++s;

    ptr = s;
    return true;
}

//================================================================
//
// skipText
//
// Parses and skips specified text literally.
//
// Returns an indication of successful parsing.
// If parsing is not successful, does not move the pointer.
//
//================================================================

template <typename Iterator, typename TextIterator>
sysinline bool skipText(Iterator& strPtr, Iterator strEnd, TextIterator textPtr, TextIterator textEnd)
{
    auto ptr = strPtr;

    while (ptr != strEnd && textPtr != textEnd && *ptr == *textPtr)
        {++ptr; ++textPtr;}

    if_not (textPtr == textEnd) // skipped to the end?
        return false;

    strPtr = ptr;
    return true;
}

//----------------------------------------------------------------

template <typename Iterator, typename Text>
sysinline bool skipText(Iterator& strPtr, Iterator strEnd, const Text& text)
    {return skipText(strPtr, strEnd, text.ptr, text.ptr + text.size);}

//================================================================
//
// skipTextThenSpaceTab
//
//================================================================

template <typename Iterator, typename Text>
sysinline bool skipTextThenSpaceTab(Iterator& strPtr, Iterator strEnd, const Text& text)
{
    ensure(skipText(strPtr, strEnd, text));
    skipSpaceTab(strPtr, strEnd);
    return true;
}

//================================================================
//
// skipTextThenAnySpace
//
//================================================================

template <typename Iterator, typename Text>
sysinline bool skipTextThenAnySpace(Iterator& strPtr, Iterator strEnd, const Text& text)
{
    ensure(skipText(strPtr, strEnd, text));
    skipAnySpace(strPtr, strEnd);
    return true;
}

//================================================================
//
// getNextLine
//
// Gets next line of a character array.
// Returns false if the pointer cannot be advanced.
//
//================================================================

template <typename Iterator>
sysinline bool getNextLine(Iterator& ptr, Iterator end, Iterator& resultBeg, Iterator& resultEnd)
{
    auto s = ptr;

    resultBeg = s;

    while (s != end && *s != '\r' && *s != '\n')
        ++s;

    resultEnd = s;

    ////

    if (s != end && *s == '\r')
        ++s;

    if (s != end && *s == '\n')
        ++s;

    ////

    bool changed = (s != ptr);
    ptr = s;
    return changed;
}

//================================================================
//
// skipUint
//
//================================================================

template <typename Iterator>
sysinline bool skipUint(Iterator& ptr, Iterator end)
{
    auto s = ptr;

    while (s != end && isDigit(*s))
        ++s;

    ensure(s != ptr);

    ptr = s;
    return true;
}

//================================================================
//
// skipInt
//
//================================================================

template <typename Iterator>
sysinline bool skipInt(Iterator& ptr, Iterator end, bool allowPlus)
{
    auto s = ptr;

    ensure(ptr != end);

    if (*s == '-')
        ++s;
    else if (*s == '+' && allowPlus)
        ++s;

    ensure(skipUint(s, end));

    ptr = s;
    return true;
}

//================================================================
//
// skipFloat
//
// If parsing is not successful, does not move the pointer.
//
//================================================================

template <typename Iterator>
sysinline bool skipFloat(Iterator& ptr, Iterator end, bool allowBodyPlus)
{
    auto s = ptr;

    ////

    ensure(skipInt(s, end, allowBodyPlus));

    ////

    if (s != end && *s == '.')
    {
        ++s;

        ensure(skipUint(s, end));
    }

    ////

    if (s != end && (*s == 'E' || *s == 'e'))
    {
        ++s;

        ensure(skipInt(s, end, true));
    }

    ////

    ptr = s;
    return true;
}

//================================================================
//
// decodeJsonStr
//
// Parses and decodes JSON string literal.
// If parsing is not successful, does not move the pointer.
//
//================================================================

template <typename Iterator, typename Writer>
sysinline bool decodeJsonStr(Iterator& ptr, Iterator end, Writer& writer)
{
    auto s = ptr;

    ////

    constexpr auto quote = '"';

    ////

    ensure(s != end && *s == quote);
    ++s;

    ////

    for (;;)
    {
        auto scanStart = s;

        while (s != end && *s != '\\' && *s != quote)
            ++s;

        ensure(s != end);

        writer(scanStart, s - scanStart);

        if (*s == quote)
            break;

        ////

        ++s;
        ensure(s != end);

        auto c = *s++;
        auto dstChar = c;

        ////

        if (c == '"' || c == '\\' || c == '/')
        {
        }

        #define TMP_MACRO(src, dst) \
            else if (c == (src)) dstChar = (dst);

        TMP_MACRO('n', '\n')
        TMP_MACRO('r', '\r')
        TMP_MACRO('t', '\t')
        TMP_MACRO('b', '\b')
        TMP_MACRO('f', '\f')

        #undef TMP_MACRO

        else if (c == 'u')
        {
            unsigned result = 0;

            for_count (i, 4)
                ensure(s != end && readAccumHexDigit(*s++, result));

            using Char = decltype(dstChar);
            dstChar = Char(result);
        }

        writer(&dstChar, 1);
    }

    ////

    ensure(s != end && *s == quote);
    ++s;

    ptr = s;
    return true;
}

//================================================================
//
// skipJsonStr
//
//================================================================

template <typename Iterator>
sysinline bool skipJsonStr(Iterator& ptr, Iterator end)
{
    auto writer = [&] (auto* ptr, auto size) {};
    return decodeJsonStr(ptr, end, writer);
}

//================================================================
//
// encodeJsonStr
//
// Encodes JSON string. Does not include surrounding quotes.
//
//================================================================

template <typename Iterator, typename Writer>
sysinline void encodeJsonStr(Iterator ptr, Iterator end, Writer& writer)
{
    auto s = ptr;

    for (;;)
    {
        //
        // The only characters that MUST be escaped are \, ", and anything less than U+0020.
        //

        auto scanStart = s;

        while (s != end && *s != '\\' && *s != '"' && unsigned(*s) >= 0x20)
            ++s;

        writer(scanStart, s - scanStart);

        if (s == end)
            break;

        ////

        auto c = *s++;

        using Char = decltype(c);
        Char dstChar = 0;

        {
            if (unsigned(c) >= 0x20)
                dstChar = c;

            #define TMP_MACRO(dst, src) \
                else if (c == (src)) dstChar = (dst);

            TMP_MACRO('n', '\n')
            TMP_MACRO('r', '\r')
            TMP_MACRO('t', '\t')
            TMP_MACRO('b', '\b')
            TMP_MACRO('f', '\f')

            #undef TMP_MACRO
        }

        if (dstChar != 0)
        {
            Char buffer[] = {'\\', dstChar};
            writer(buffer, COMPILE_ARRAY_SIZE(buffer));
            continue;
        }

        ////

        {
            Char buffer[6];
            auto bufPtr = buffer + COMPILE_ARRAY_SIZE(buffer);

            using UChar = TYPE_MAKE_UNSIGNED(Char);
            auto value = unsigned(UChar(c));

            for_count (i, 4)
            {
                *--bufPtr = "0123456789ABCDEF"[value & 0xF];
                value >>= 4;
            }

            *--bufPtr = 'u';
            *--bufPtr = '\\';

            writer(buffer, COMPILE_ARRAY_SIZE(buffer));
        }
    }
}
