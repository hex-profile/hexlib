#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"
#include "storage/constructDestruct.h"

namespace fastAllocator {

//================================================================
//
// The module implements fast stack-based allocators.
//
// The allocators use:
//
// * AddrU address type for allocation work. The type is >= SpaceU.
// * SpaceU type for memory block size and alignment.
//
//================================================================

//================================================================
//
// FastAllocator
//
// * Allocations and deallocations are made by stack principle
// (with proper alignment).
//
// * If user code tries to deallocate NOT the last allocated block,
// the allocator locks in error state and refuses further allocations.
//
// The class has two modes: counting mode and real alloc mode
// (compile-time option).
//
// In counting mode, the alignment of a request can be arbitrarily big,
// and max used alignment is recorded.
//
// In compile-time "state mode" deallocations are ignored and owner is not set.
//
//================================================================

//================================================================
//
// FastAllocatorState
//
//================================================================

template <typename AddrU>
class FastAllocatorState
{

    template <typename, bool, bool>
    friend class FastAllocator;

public:

    sysinline FastAllocatorState()
        :
        memAddr(0),
        memSize(0)
    {
        offset = 0;
        maxAlignment = 1;
        maxOffset = 0;
    }

    sysinline FastAllocatorState(AddrU memAddr, AddrU memSize)
        :
        memAddr(memAddr),
        memSize(memSize)
    {
        offset = 0;
        maxAlignment = 1;
        maxOffset = 0;
    }

private:

    // Flag indicating the absense of stack order violations.
    // If cleared, all subsequent allocations are refused.
    bool validState = true;

    AddrU offset;
    AddrU maxAlignment;
    AddrU maxOffset;

    AddrU memAddr;
    AddrU memSize;

};

//================================================================
//
// FastAllocator
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
class FastAllocator : public AllocatorInterface<AddrU>
{

public:

    sysinline FastAllocator(const ErrorLogKit& kit)
        : kit(kit) {}

    sysinline FastAllocator(AddrU memAddr, AddrU memSize, const ErrorLogKit& kit)
        : state(memAddr, memSize), kit(kit) {}

public:

    stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars);
    static void deallocFunc(MemoryDeallocContext& context);

public:

    bool validState() const
        {return state.validState;}

    AddrU allocatedSpace() const
        {return state.offset;}

    AddrU maxAllocatedSpace() const
        {return state.maxOffset;}

    AddrU maxAlignment() const
        {return state.maxAlignment;}

private:

    FastAllocatorState<AddrU> state;
    ErrorLogKit kit;

};

//----------------------------------------------------------------

}
