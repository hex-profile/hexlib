#include "fastAllocator.h"

#include "numbers/int/intType.h"
#include "errorLog/debugBreak.h"

namespace fastAllocator {

//================================================================
//
// FastAllocatorDeallocContext
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
struct FastAllocatorDeallocContext
{
    FastAllocator<AddrU, realAlloc, stateMode>* allocator;
    AddrU restoreValue;
    AddrU checkValue;
};

//================================================================
//
// FastAllocator::allocFunc
//
// ~55-57 instructions on X86 (without allocation curve check).
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
stdbool FastAllocator<AddrU, realAlloc, stateMode>::alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull)
{
    // State?
    REQUIRE(validState);

    // Alignment.
    REQUIRE(isPower2(alignment));
    REQUIRE(alignment >= 1);

    AddrU alignmentMask = alignment - 1;

    // Round the offset UP to a multiple of the alignment.
    AddrU originalOffset = offset;
    REQUIRE(originalOffset <= TYPE_MAX(AddrU) - alignmentMask); // can add
    AddrU alignedOffset = (originalOffset + alignmentMask) & ~alignmentMask;

    // Add required size.
    AddrU sizeU = size;
    REQUIRE(alignedOffset <= TYPE_MAX(AddrU) - sizeU);
    AddrU newOffset = alignedOffset + sizeU;

    // Fits into the memory?
    if (realAlloc)
    {
        REQUIRE_CUSTOM(newOffset <= memSize, CT("Insufficient memory."));
        REQUIRE_CUSTOM((memAddr & alignmentMask) == 0, CT("Insufficient alignment of system buffer."));
    }

    // Allocation curve checker.
    if_not (realAlloc)
    {
        if (curvePtr != curveEnd)
            *curvePtr++ = newOffset;
        else
        {
            CHECK_CUSTOM(curveReported, CT("Allocation curve checker capacity exceeded."));
            curveReported = true;
        }
    }
    else
    {
        if_not (curvePtr != curveEnd && *curvePtr++ == newOffset)
        {
            CHECK_CUSTOM(curveReported, CT("Allocation curve mismatch between counting and execution phases."));
            curveReported = true;
        }
    }

    // Record changes.
    offset = newOffset;

    // Accumulate counters
    maxAlignment = maxv(maxAlignment, alignment);
    maxOffset = maxv(maxOffset, newOffset);

    // Deallocator.
    if (stateMode)
        owner.clear();
    else
    {
        MemoryDeallocContext& deallocContext = owner.replace(deallocFunc);

        auto& info = deallocContext.recast<FastAllocatorDeallocContext<AddrU, realAlloc, stateMode>>();

        info.allocator = this;
        info.restoreValue = originalOffset;
        info.checkValue = newOffset;
    }

    // Result.
    AddrU allocatedAddr = memAddr + alignedOffset;
    result = realAlloc ? allocatedAddr : 0;

    returnTrue;
}

//================================================================
//
// FastAllocator::deallocFunc
//
// ~7 instructions on X86.
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
void FastAllocator<AddrU, realAlloc, stateMode>::deallocFunc(MemoryDeallocContext& context)
{
    auto& info = (FastAllocatorDeallocContext<AddrU, realAlloc, stateMode>&) context;
    auto& the = *info.allocator;

    if (the.offset == info.checkValue)
        the.offset = info.restoreValue;
    else
    {
        // Lock error state and refuse subsequent allocations.
        DEBUG_BREAK_INLINE();
        the.validState = false;
    }
}

//================================================================
//
// FastAllocator instantiations
//
//================================================================

#define TMP_MACRO(AddrU) \
    template class FastAllocator<AddrU, false, false>; \
    template class FastAllocator<AddrU, false, true>; \
    template class FastAllocator<AddrU, true, false>; \
    template class FastAllocator<AddrU, true, true>; \

TMP_MACRO(size_t)

#undef TMP_MACRO

//----------------------------------------------------------------

}
