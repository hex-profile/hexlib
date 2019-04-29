#pragma once

#include <string.h>

//================================================================
//
// trivialEq
//
//================================================================

template <typename Type>
inline bool trivialEq(const Type& A, const Type& B)
{
    return memcmp(&A, &B, sizeof(Type)) == 0;
}
