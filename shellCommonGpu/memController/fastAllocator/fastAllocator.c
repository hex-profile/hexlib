#include "fastAllocator.h"

#include "numbers/int/intType.h"
#include "errorLog/debugBreak.h"

namespace fastAllocator {

//================================================================
//
// FastAllocatorDeallocContext
//
//================================================================

template <typename AddrU>
struct FastAllocatorDeallocContext
{
    void* allocState;
    AddrU restoreValue;
    AddrU checkValue;
};

//================================================================
//
// FastAllocator::allocFunc
//
// ~55-57 instructions on X86.
//
//================================================================

template <typename AddrU, bool realAlloc, bool stateMode>
stdbool FastAllocator<AddrU, realAlloc, stateMode>::alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
{
    auto& that = state;

    // State?
    REQUIRE(that.validState);

    // Alignment.
    REQUIRE(isPower2(alignment));
    REQUIRE(alignment >= 1);

    AddrU alignmentMask = alignment - 1;

    // Round the offset UP to a multiple of the alignment.
    AddrU originalOffset = that.offset;
    REQUIRE(originalOffset <= TYPE_MAX(AddrU) - alignmentMask); // can add
    AddrU alignedOffset = (originalOffset + alignmentMask) & ~alignmentMask;

    // Add required size.
    AddrU sizeU = size;
    REQUIRE(alignedOffset <= TYPE_MAX(AddrU) - sizeU);
    AddrU newOffset = alignedOffset + sizeU;

    // Fits into the memory?
    if (realAlloc)
    {
        REQUIRE(newOffset <= that.memSize);
        REQUIRE((that.memAddr & alignmentMask) == 0);
    }

    // Record "that" changes.
    that.offset = newOffset;

    // Accumulate counters
    that.maxAlignment = maxv(that.maxAlignment, alignment);
    that.maxOffset = maxv(that.maxOffset, newOffset);

    // Deallocator.
    if (stateMode)
        owner.clear();
    else
    {
        MemoryDeallocContext& deallocContext = owner.replace(deallocFunc);

        auto& info = deallocContext.recast<FastAllocatorDeallocContext<AddrU>>();

        info.allocState = &state;
        info.restoreValue = originalOffset;
        info.checkValue = newOffset;
    }

    // Result.
    AddrU allocatedAddr = that.memAddr + alignedOffset;
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
    FastAllocatorDeallocContext<AddrU>& info = (FastAllocatorDeallocContext<AddrU>&) context;

    FastAllocatorState<AddrU>& state = * (FastAllocatorState<AddrU>*) info.allocState;

    if (state.offset == info.checkValue)
        state.offset = info.restoreValue;
    else
    {
        // Lock error state and refuse subsequent allocations.
        DEBUG_BREAK_INLINE();
        state.validState = false;
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
