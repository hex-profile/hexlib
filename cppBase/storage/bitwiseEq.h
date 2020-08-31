#pragma once

#include <string.h>

//================================================================
//
// bitwiseEq
//
//================================================================

template <typename Type>
inline bool bitwiseEq(const Type& A, const Type& B)
{
    return memcmp(&A, &B, sizeof(Type)) == 0;
}
