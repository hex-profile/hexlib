#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// TYPEOF
// TYPEOF_REGISTER
//
// The interface of typeof.
//
//================================================================

//================================================================
//
// TypeofID
//
//================================================================

using TypeofID = size_t;

//================================================================
//
// Typeof_SizeType
//
//================================================================

template <TypeofID id>
struct Typeof_SizeType
{
    struct T {char data[id];};
};

//----------------------------------------------------------------

COMPILE_ASSERT(sizeof(Typeof_SizeType<1>::T) == 1);
COMPILE_ASSERT(sizeof(Typeof_SizeType<5>::T) == 5);

//================================================================
//
// Typeof_RetrieveType
//
//================================================================

template <TypeofID id>
struct Typeof_RetrieveType;

//================================================================
//
// TYPEOF_REGISTER
//
//================================================================

#define TYPEOF_REGISTER(id, Type) \
    \
    Typeof_SizeType<id>::T typeofDetermineType_(Type* value); \
    \
    template <> \
    struct Typeof_RetrieveType<id> \
    { \
        using T = Type; \
    };

//================================================================
//
// typeof_Purify
//
//================================================================

template <typename Type>
Type* typeof_Purify(const volatile Type& value);

//================================================================
//
// TYPEOF
//
//================================================================

#define TYPEOF(expr) \
    Typeof_RetrieveType<sizeof(typeofDetermineType_(typeof_Purify(expr)))>::T
