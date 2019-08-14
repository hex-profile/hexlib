#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"

//================================================================
//
// skipSpaceTab
//
//================================================================

template <typename Iterator>
inline bool skipSpaceTab(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

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
inline bool skipAnySpace(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

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
inline bool skipNonSpaceCharacters(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

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
// Parses a C/C++ identifier
//
// Returns an indication of successful parsing
// if parsing is not successful, does not move pointer.
//
//================================================================

template <typename Iterator>
inline bool skipIdent(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

    ensure(s != end && isIdent1st(*s));

    ++s;

    while (s != end && isIdentNext(*s))
        ++s;

    ptr = s;
    return true;
}

//================================================================
//
// skipCppComment
//
// Parses C++ comment.
//
// Returns an indication of successful parsing
// if parsing is not successful, does not move pointer.
//
//================================================================

template <typename Iterator>
inline bool skipCppComment(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

    ensure(s != end && *s == '/');
    ++s;

    ensure(s != end && *s == '/');
	++s;

    s = end;

    ptr = s;
    return true;
}

//================================================================
//
// skipCstr
//
// Parses and skips C/C++ string literal like "..." or '...'.
//
// returns an indication of successful parsing
// if parsing is not successful, does not move pointer
//
//================================================================

template <typename Iterator>
inline bool skipCstr(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

    ////

    ensure(s != end && (*s == '"' || *s == '\''));
    auto quote = *s;
    ++s;

    ////

    bool slash = false;

    while (s != end && (*s != quote || slash))
    {
        slash = !slash && (*s == '\\');
        ++s;
    }

    ensure(s != end && *s == quote);
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
// If parsing is not successful, it does not move the pointer.
//
//================================================================

template <typename Iterator, typename TextIterator>
inline bool skipText
(
    Iterator& strPtr,
    Iterator strEnd,
    TextIterator textPtr,
    TextIterator textEnd
)
{
    Iterator ptr = strPtr;

    while (ptr != strEnd && textPtr != textEnd && *ptr == *textPtr)
        {++ptr; ++textPtr;}

    if_not (textPtr == textEnd) // skipped to the end?
        return false;

    strPtr = ptr;
    return true;
}

//----------------------------------------------------------------

template <typename Iterator, typename Text>
inline bool skipText(Iterator& strPtr, Iterator strEnd, const Text& text)
    {return skipText(strPtr, strEnd, text.ptr, text.ptr + text.size);}

//================================================================
//
// skipTextThenSpaceTab
//
//================================================================

template <typename Iterator, typename Text>
inline bool skipTextThenSpaceTab(Iterator& strPtr, Iterator strEnd, const Text& text)
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
inline bool skipTextThenAnySpace(Iterator& strPtr, Iterator strEnd, const Text& text)
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
inline bool getNextLine(Iterator& ptr, Iterator end, Iterator& resultBeg, Iterator& resultEnd)
{
    Iterator s = ptr;

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
inline bool skipUint(Iterator& ptr, Iterator end)
{
    Iterator s = ptr;

    while (s != end && isDigit(*s))
        ++s;

    ensure(s != ptr);

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

    for (; s != end && isDigit(*s); ++s)
        value = value * 10 + Uint(*s - '0');

    ensure(s != ptr);

    ptr = s;
    result = value;
    return true;
}

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
