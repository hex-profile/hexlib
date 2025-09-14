#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"
#include "storage/constructDestruct.h"
#include "data/array.h"

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
// FastAllocator
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
class FastAllocator : public AllocatorInterface<AddrU>
{

public:

    sysinline FastAllocator(const Array<AddrU>& curveBuffer, const ErrorLogKit& kit)
        : kit(kit)
    {
        curveSetBuffer(curveBuffer);
    }

    sysinline FastAllocator(AddrU memAddr, AddrU memSize, const Array<AddrU>& curveBuffer, const ErrorLogKit& kit)
        : memAddr(memAddr), memSize(memSize), kit(kit)
    {
        curveSetBuffer(curveBuffer);
    }

    sysinline void curveSetBuffer(const Array<AddrU>& buffer)
    {
        ARRAY_EXPOSE_UNSAFE(buffer);
        curveOrg = bufferPtr;
        curveEnd = bufferPtr + bufferSize;
        curvePtr = bufferPtr;
        curveReported = (curveOrg == curveEnd);
    }

public:

    void alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull);
    static void deallocFunc(MemoryDeallocContext& context);

public:

    sysinline bool isValid() const
        {return validState;}

    sysinline AddrU allocatedSpace() const
        {return offset;}

    sysinline AddrU maxAllocatedSpace() const
        {return maxOffset;}

    sysinline AddrU maxAlign() const
        {return maxAlignment;}

    sysinline Space curveSize() const
        {return curvePtr - curveOrg;}

    sysinline bool curveIsReported() const
        {return curveReported;}

private:

    // Flag indicating the absense of stack order violations.
    // If cleared, all subsequent allocations are refused.
    bool validState = true;

    AddrU offset = 0;
    AddrU maxAlignment = 1;
    AddrU maxOffset = 0;

    AddrU memAddr = 0;
    AddrU memSize = 0;

    AddrU* curveOrg = nullptr;
    AddrU* curveEnd = nullptr;
    AddrU* curvePtr = nullptr;
    bool curveReported = false;

    ErrorLogKit kit;

};

//----------------------------------------------------------------

}
