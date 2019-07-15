#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"
#include "numbers/int/intType.h"

//================================================================
//
// sysAllocAlignShell
//
//================================================================

template <typename AddrU, typename CoreAlloc, typename Owner, typename Kit, typename DeallocFunc>
static inline stdbool sysAllocAlignShell(AddrU size, AddrU alignment, Owner& owner, AddrU& result, CoreAlloc& coreAlloc, DeallocFunc* deallocFunc, stdPars(Kit))
{
    REQUIRE(isPower2(alignment));
    REQUIRE(alignment >= 1);

    AddrU alignMask = alignment - 1;

    ////

    AddrU maxCorrection = alignMask;
    REQUIRE(size <= TYPE_MAX(SpaceU) - maxCorrection);
    AddrU allocSize = size + maxCorrection;

    ////

    AddrU allocPtr = 0;
    require(coreAlloc(allocPtr, allocSize, stdPass));

    ////

    AddrU alignedPtr = (allocPtr + alignMask) & ~alignMask; // overflow impossible

    ////

    MemoryDeallocContext& context = owner.replace(deallocFunc);

    context.recast<AddrU>() = allocPtr;

    result = alignedPtr;

    returnTrue;
}
