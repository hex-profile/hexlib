#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "allocation/flatMemoryAllocator.h"
#include "errorLog/errorLog.h"
#include "numbers/interface/numberInterface.h"
#include "errorLog/debugBreak.h"

//================================================================
//
// FlatToSpaceAllocatorThunk
//
// Makes implementation of AllocatorObject based on the provided flat allocator.
//
//================================================================

template <typename AddrU>
class FlatToSpaceAllocatorThunk : public AllocatorInterface<AddrU>, public BlockAllocatorInterface<AddrU>
{

private:

    stdbool alloc(AllocatorState& self, AddrU size, SpaceU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
        {return flatAllocator.alloc(size, alignment, owner, result, stdNullPassThru);}

public:

    FlatToSpaceAllocatorThunk(FlatMemoryAllocator<AddrU>& flatAllocator, const ErrorLogKit& kit)
        : flatAllocator(flatAllocator), kit(kit) {}

public:

    void initCountingState(AllocatorState& state) {DEBUG_BREAK_INLINE();}
    void initDistribState(AllocatorState& state, AddrU memAddr, AddrU memSize) {DEBUG_BREAK_INLINE();}

    virtual bool validState(const AllocatorState& state) {DEBUG_BREAK_INLINE(); return true;}
    virtual AddrU allocatedSpace(const AllocatorState& state) {DEBUG_BREAK_INLINE(); return 0;}
    virtual AddrU maxAllocatedSpace(const AllocatorState& state) {DEBUG_BREAK_INLINE(); return 0;}
    virtual SpaceU maxAlignment(const AllocatorState& state) {DEBUG_BREAK_INLINE(); return 0;}

private:

    FlatMemoryAllocator<AddrU>& flatAllocator;
    ErrorLogKit kit;

};
