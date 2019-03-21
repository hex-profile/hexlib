#pragma once

#include "stdFunc/stdFunc.h"
#include "storage/opaqueStruct.h"
#include "dataAlloc/deallocInterface.h"
#include "kit/kit.h"
#include "dataAlloc/memoryAllocatorKit.h"
#include "data/space.h"

//================================================================
//
// AllocatorState
//
//================================================================

using AllocatorState = OpaqueStruct<64>;

//================================================================
//
// MemoryOwner
//
//================================================================

using MemoryDeallocContext = DeallocContext<24, 0x4C0E823E>;

////

class MemoryOwner : public ResourceOwner<MemoryDeallocContext>
{
    inline friend void exchange(MemoryOwner& a, MemoryOwner& b)
    {
        ResourceOwner<MemoryDeallocContext>* ap = &a;
        ResourceOwner<MemoryDeallocContext>* bp = &b;

        exchange(*ap, *bp);
    }
};

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
    virtual bool alloc(AllocatorState& state, AddrU size, SpaceU alignment, MemoryOwner& owner, AddrU& result, stdNullPars) =0;
};

//================================================================
//
// Advanced API for block allocators
//
//================================================================

template <typename AddrU>
struct BlockAllocatorInterface
{
    virtual void initCountingState(AllocatorState& state) =0;
    virtual void initDistribState(AllocatorState& state, AddrU memAddr, AddrU memSize) =0;

    virtual bool validState(const AllocatorState& state) =0;
    virtual AddrU allocatedSpace(const AllocatorState& state) =0;
    virtual AddrU maxAllocatedSpace(const AllocatorState& state) =0;
    virtual SpaceU maxAlignment(const AllocatorState& state) =0;
};

//================================================================
//
// AllocatorObject
//
//================================================================

template <typename AddrU>
KIT_CREATE2_(AllocatorObject, AllocatorState&, state, AllocatorInterface<AddrU>&, func);
