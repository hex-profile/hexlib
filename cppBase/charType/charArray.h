#pragma once

#include "charType/charType.h"
#include "compileTools/compileTools.h"

#include <string.h>

//================================================================
//
// CharType array definitions.
//
// A pair of (pointer, size) is more efficient than null-terminated strings.
//
//================================================================

//================================================================
//
// CharArrayEx
//
//================================================================

template <typename Type>
class CharArrayEx
{

public:

    sysinline CharArrayEx(int=0)
        : size(0) {}

    sysinline CharArrayEx(const Type* ptr, size_t size)
        : ptr(ptr), size(size) {}

public:

    const Type* ptr;
    size_t size;

};

//----------------------------------------------------------------

using CharArray = CharArrayEx<CharType>;

//================================================================
//
// CHARARRAY_LITERAL
// CHARARRAY_STATIC
//
// Macros making char array from string literals,
// like MS "_T" macros for null-terminated strings.
//
// Garbage character is added behind the end of string
// to better debug null-terminated string expectancies.
//
//================================================================

#define CHARARRAY_LITERAL(x) \
    CharArray(CT(x) CT("@"), COMPILE_ARRAY_SIZE(x) - 1)

#define CHARARRAY_STATIC(x) \
    {CT(x) CT("@"), COMPILE_ARRAY_SIZE(x) - 1}

// Ensure that compiler computes it correctly.
COMPILE_ASSERT(COMPILE_ARRAY_SIZE(CT("1234")) == 5);

//================================================================
//
// STR
//
// Shorter names for convenience.
//
// STR is for string-literals like STR("string").
//
//================================================================

#define STR(x) \
    CHARARRAY_LITERAL(x)

//================================================================
//
// charArrayFromPtr
//
// Interop with C string, not to be used often.
//
//================================================================

template <typename Type>
CharArrayEx<Type> charArrayFromPtr(const Type* cstring);

//================================================================
//
// strEqual
//
//================================================================

template <typename Type>
sysinline bool strEqual(const CharArrayEx<Type>& a, const CharArrayEx<Type>& b)
{
    return
        a.size == b.size &&
        memcmp(a.ptr, b.ptr, a.size * sizeof(Type)) == 0;
}
