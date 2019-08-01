#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"
#include "charType/charArray.h"

//================================================================
//
// skipSpaceTab
//
//================================================================

template <typename Char>
inline bool skipSpaceTab(const Char*& ptr, const Char* end)
{
    const Char* s = ptr;

    while (s != end && isSpaceTab(*s))
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

template <typename Char>
inline bool skipNonSpaceCharacters(const Char*& ptr, const Char* end)
{
    const Char* s = ptr;

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

template <typename Char>
inline bool skipIdent(const Char*& ptr, const Char* end)
{
    const Char* s = ptr;

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

template <typename Char>
inline bool skipCppComment(const Char*& ptr, const Char* end)
{
    const Char* s = ptr;

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

template <typename Char>
inline bool skipCstr(const Char*& ptr, const Char* end)
{
    const Char* s = ptr;

    ////

    ensure(s != end && (*s == '"' || *s == '\''));
    Char quote = *s;
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

template <typename Char, typename TextChar>
inline bool skipText
(
    const Char*& strPtr,
    const Char* strEnd,
    const TextChar* textPtr,
    const TextChar* textEnd
)
{
    const Char* ptr = strPtr;

    while (ptr != strEnd && textPtr != textEnd && *ptr == *textPtr)
        {++ptr; ++textPtr;}

    if_not (textPtr == textEnd) // skipped to the end?
        return false;

    strPtr = ptr;
    return true;
}

//----------------------------------------------------------------

template <typename Char, typename TextChar>
inline bool skipText(const Char*& strPtr, const Char* strEnd, const CharArrayEx<TextChar>& text)
    {return skipText(strPtr, strEnd, text.ptr, text.ptr + text.size);}

//================================================================
//
// skipTextThenSpace
//
//================================================================

template <typename Char, typename TextChar>
inline bool skipTextThenSpace(const Char*& strPtr, const Char* strEnd, const CharArrayEx<TextChar>& text)
{
    ensure(skipText(strPtr, strEnd, text));
    skipSpaceTab(strPtr, strEnd);
    return true;
}

//================================================================
//
// getNextLine
//
// Gets next line of a char array.
// Returns false if the pointer cannot be advanced.
//
//================================================================

template <typename Char>
inline bool getNextLine(const Char*& ptr, const Char* end, const Char*& resultBeg, const Char*& resultEnd)
{
    const Char* originalPtr = ptr;

    resultBeg = ptr;

    while (ptr != end && !isNewLine(*ptr))
        ++ptr;

    resultEnd = ptr;

    while (ptr != end && isNewLine(*ptr))
        ++ptr;

    return ptr != originalPtr;
}

//================================================================
//
// readUint
//
//================================================================

template <typename Char, typename Uint>
inline bool readUint(const Char*& ptr, const Char* end, Uint& result)
{
    const Char* s = ptr;

    Uint value = 0;

    for (; s != end && isDigit(*s); ++s)
        value = value * 10 + Uint(*s - '0');

    ensure(s != ptr);

    ptr = s;
    result = value;
    return true;
}
