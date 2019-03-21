#pragma once

#ifndef HEXLIB_OPAQUE_STRUCT
#define HEXLIB_OPAQUE_STRUCT

#include <type_traits>
#include <stddef.h>

//================================================================
//
// OpaqueStruct
//
// The structure with untyped memory of specified size.
// The memory has alignment suitable for any built-in type.
//
//================================================================

template <size_t size>
struct OpaqueStruct
{
    std::aligned_storage_t<size> data; // uses default alignment
};

//================================================================
//
// exchange<OpaqueStruct>
//
//================================================================

template <size_t size>
inline void exchange(OpaqueStruct<size>& a, OpaqueStruct<size>& b)
{
    OpaqueStruct<size> tmp = a;
    a = b;
    b = tmp;
}

//----------------------------------------------------------------

#endif // HEXLIB_OPAQUE_STRUCT
