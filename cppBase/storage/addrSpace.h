#pragma once

#include "compileTools/compileTools.h"

//================================================================
//
// Byte
//
//----------------------------------------------------------------
//
// Minimal address unit.
//
// To use with typeless memory blocks, that is where
// void*, size_t and ptrdiff_t are used.
//
// Byte is guaranteed to be UNSIGNED integer type.
//
//================================================================

using Byte = unsigned char;

COMPILE_ASSERT(sizeof(Byte) == 1);

//================================================================
//
// CpuAddrS
// CpuAddrU
//
// C++ language address space.
//
//================================================================

using CpuAddrS = ptrdiff_t;
using CpuAddrU = size_t;

COMPILE_ASSERT(sizeof(CpuAddrU) == sizeof(void*));
COMPILE_ASSERT(sizeof(CpuAddrS) == sizeof(void*));

//================================================================
//
// addOffset
//
//================================================================

template <typename Type>
sysinline Type* addOffset(Type* ptr, ptrdiff_t ofs)
{
    return (Type*) (((Byte*) ptr) + ofs);
}
