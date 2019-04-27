#pragma once

#include "allocation/flatMemoryAllocator.h"
#include "errorLog/errorLogKit.h"
#include "allocation/flatToSpaceAllocatorThunk.h"

//================================================================
//
// MallocFlatAllocatorThunk
//
// Malloc-based implementation of FlatMemoryAllocator.
//
//================================================================

template <typename AddrU>
class MallocFlatAllocatorThunk : public FlatMemoryAllocator<AddrU>
{

public:

    stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars);

    static void dealloc(MemoryDeallocContext& context);

public:

    inline MallocFlatAllocatorThunk(const ErrorLogKit& kit) : kit(kit) {}

private:

    ErrorLogKit kit;

};

//================================================================
//
// MAKE_MALLOC_ALLOCATOR_OBJECT
//
//================================================================

#define MAKE_MALLOC_ALLOCATOR_OBJECT(kit) \
    MallocFlatAllocatorThunk<CpuAddrU> mallocFlat(kit); \
    FlatToSpaceAllocatorThunk<CpuAddrU> mallocSpace(mallocFlat, kit); \
    AllocatorState mallocUnusedState; \
    AllocatorObject<CpuAddrU> mallocAllocator(mallocUnusedState, mallocSpace);
