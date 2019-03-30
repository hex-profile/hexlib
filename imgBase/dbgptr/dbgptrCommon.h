#pragma once

#include "compileTools/compileTools.h"
#include "data/space.h"

//================================================================
//
// DbgptrAddrU
//
// The unsigned address space type.
// If the memory model is not flat, debug pointers refactoring is required.
//
// The type should have modulo 2^n arithmetic, the same as for built-in pointers,
// otherwise debug pointers refactoring is required.
//
//================================================================

using DbgptrAddrU = size_t;
COMPILE_ASSERT(sizeof(DbgptrAddrU) == sizeof(void*));
COMPILE_ASSERT(sizeof(DbgptrAddrU) >= sizeof(SpaceU));

using DbgptrAddrS = ptrdiff_t;
COMPILE_ASSERT(sizeof(DbgptrAddrS) == sizeof(void*));
COMPILE_ASSERT(sizeof(DbgptrAddrU) >= sizeof(Space));
