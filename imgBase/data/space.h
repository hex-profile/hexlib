#pragma once

#ifndef HEXLIB_SPACE
#define HEXLIB_SPACE

#include <cstdint>

//================================================================
//
// Space
//
// The address space type of data containers.
// Space is signed integer address space type, similar to ptrdiff_t.
//
// The type can hold any data container size in bytes,
// and signed difference between pointers to any two elements of the same data container.
//
// Space type can be less than C++ address space.
// For example, on a 64-bit system, memory allocation is performed in 64-bit
// address space, but any single container is limited to 2GB.
//
//================================================================

using Space = int32_t;
static const Space spaceMax = 0x7FFFFFFF;

using SpaceU = uint32_t;
static_assert(sizeof(SpaceU) == sizeof(Space), "");
static_assert(sizeof(SpaceU) <= sizeof(size_t), "");

//----------------------------------------------------------------

#endif
