#pragma once

#include "parseTools/charSet.h"
#include "compileTools/compileTools.h"
#include "charType/charArray.h"

//================================================================
//
// skipSpaceTab
//
//================================================================

inline bool skipSpaceTab(const CharType*& ptr, const CharType* end)
{
    const CharType* s = ptr;

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

inline bool skipNonSpaceCharacters(const CharType*& ptr, const CharType* end)
{
    const CharType* s = ptr;

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

inline bool skipIdent(const CharType*& ptr, const CharType* end)
{
    const CharType* s = ptr;

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

inline bool skipCppComment(const CharType*& ptr, const CharType* end)
{
    const CharType* s = ptr;

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

inline bool skipCstr(const CharType*& ptr, const CharType* end)
{
    const CharType* s = ptr;

    ////

    ensure(s != end && (*s == '"' || *s == '\''));
    CharType quote = *s;
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
// Returns an indication of successful parsing
// If parsing is not successful, does not move pointer.
//
//================================================================

inline bool skipText
(
    const CharType*& strPtr,
    const CharType* strEnd,
    const CharType* textPtr,
    const CharType* textEnd
)
{
    const CharType* ptr = strPtr;

    while (ptr != strEnd && textPtr != textEnd && *ptr == *textPtr)
        {++ptr; ++textPtr;}

    if_not (textPtr == textEnd) // skipped to the end?
        return false;

    strPtr = ptr;
    return true;
}

//----------------------------------------------------------------

inline bool skipText(const CharType*& strPtr, const CharType* strEnd, const CharArray& text)
    {return skipText(strPtr, strEnd, text.ptr, text.ptr + text.size);}

//================================================================
//
// skipTextThenSpace
//
//================================================================

inline bool skipTextThenSpace(const CharType*& strPtr, const CharType* strEnd, const CharArray& text)
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

inline bool getNextLine(const CharType*& ptr, const CharType* end, const CharType*& resultBeg, const CharType*& resultEnd)
{
    const CharType* originalPtr = ptr;

    resultBeg = ptr;

    while (ptr != end && !isNewLine(*ptr))
        ++ptr;

    resultEnd = ptr;

    while (ptr != end && isNewLine(*ptr))
        ++ptr;

    return ptr != originalPtr;
}
