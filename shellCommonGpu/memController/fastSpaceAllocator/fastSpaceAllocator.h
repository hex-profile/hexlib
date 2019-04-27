#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"
#include "storage/constructDestruct.h"

namespace fastSpaceAllocator {

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
// FastAllocatorThunk
//
// Allocator for module temporary memory:
//
// * Allocations and deallocations are made by stack principle
// (with proper alignment).
//
// * If user code tries to deallocate NOT the last allocated block,
// the allocator locks in error state and refuses further allocations.
//
// The class has two modes: counting mode and allocation mode
// (see two constructors).
//
// In counting mode, the alignment of a request can be arbitrarily big,
// and max used alignment is recorded.
//
// In "state mode" deallocations are ignored and owner is not set.
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
    friend class FastAllocatorThunk;

public:

    inline FastAllocatorState()
        :
        memAddr(0),
        memSize(0)
    {
        offset = 0;
        maxAlignment = 1;
        maxOffset = 0;
    }

    inline FastAllocatorState(AddrU memAddr, AddrU memSize)
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
    SpaceU maxAlignment;
    AddrU maxOffset;

    AddrU const memAddr;
    AddrU const memSize;

};

//================================================================
//
// FastAllocatorThunk
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
class FastAllocatorThunk : public AllocatorInterface<AddrU>, public BlockAllocatorInterface<AddrU>
{

public:

    inline FastAllocatorThunk(const ErrorLogKit& kit)
        : kit(kit) {}

public:

    stdbool alloc(AllocatorState& state, AddrU size, SpaceU alignment, MemoryOwner& owner, AddrU& result, stdNullPars);

    static void deallocFunc(MemoryDeallocContext& context);

public:

    bool validState(const AllocatorState& state)
        {return castState(state).validState;}

    AddrU allocatedSpace(const AllocatorState& state)
        {return castState(state).offset;}

    AddrU maxAllocatedSpace(const AllocatorState& state)
        {return castState(state).maxOffset;}

    SpaceU maxAlignment(const AllocatorState& state)
        {return castState(state).maxAlignment;}

public:

    void initCountingState(AllocatorState& state)
        {constructDefault(castState(state));}

    void initDistribState(AllocatorState& state, AddrU memAddr, AddrU memSize)
        {constructParams(castState(state), FastAllocatorState<AddrU>, (memAddr, memSize));}

private:

    COMPILE_ASSERT(sizeof(FastAllocatorState<AddrU>) <= sizeof(AllocatorState));

    static FastAllocatorState<AddrU>& castState(AllocatorState& state)
        {return (FastAllocatorState<AddrU>&) state;}

    static const FastAllocatorState<AddrU>& castState(const AllocatorState& state)
        {return (const FastAllocatorState<AddrU>&) state;}

private:

    ErrorLogKit kit;

};

//----------------------------------------------------------------

}
