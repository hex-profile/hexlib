#pragma once

#include "stdFunc/stdFunc.h"
#include "storage/opaqueStruct.h"
#include "dataAlloc/deallocInterface.h"
#include "kit/kit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "data/space.h"

//================================================================
//
// MemoryOwner
//
//================================================================

using MemoryDeallocContext = OpaqueStruct<24, 0x4C0E823Eu>;

////

using MemoryOwner = ResourceOwner<MemoryDeallocContext>;

//================================================================
//
// AllocatorInterface
//
// The allocator of raw memory buffers for application-level data containers.
//
// The allocator can have fast "stack" implementation.
//
//================================================================

template <typename AddrU>
struct AllocatorInterface
{
    virtual stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull) =0;
};
