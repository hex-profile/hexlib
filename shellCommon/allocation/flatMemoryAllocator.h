#pragma once

#include "dataAlloc/memoryAllocator.h"

//================================================================
//
// FlatMemoryAllocator
//
// Memory allocator interface for flat address space.
//
//================================================================

template <typename AddrU>
struct FlatMemoryAllocator
{
    virtual stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars) =0;
};
