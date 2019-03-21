#pragma once

#include "charType/charType.h"
#include "compileTools/compileTools.h"

//================================================================
//
// CharType array definitions.
//
// The project avoids using null-terminated strings,
// the pair (pointer, size) is used instead, as it's more efficient.
//
//================================================================

//================================================================
//
// CharArray
//
//================================================================

class CharArray
{

public:

    sysinline CharArray(int=0)
        : size(0) {}

    sysinline CharArray(const CharType* ptr, size_t size)
        : ptr(ptr), size(size) {}

public:

    const CharType* ptr;
    size_t size;

};

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

CharArray charArrayFromPtr(const CharType* cstring);
