#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLogKit.h"

//================================================================
//
// MallocAllocator
//
// Malloc-based implementation of AllocatorInterface.
//
//================================================================

template <typename AddrU>
class MallocAllocator : public AllocatorInterface<AddrU>
{

public:

    stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull);
    static void dealloc(MemoryDeallocContext& context);

public:

    inline MallocAllocator(const ErrorLogKit& kit) : kit(kit) {}

private:

    ErrorLogKit kit;

};

//================================================================
//
// MAKE_MALLOC_ALLOCATOR
//
//================================================================

#define MAKE_MALLOC_ALLOCATOR(kit) \
    MallocAllocator<CpuAddrU> mallocAllocator{kit}
