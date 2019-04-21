#include "mallocFlatAllocator.h"

#include "errorLog/errorLog.h"
#include "numbers/int/intType.h"
#include "allocation/sysAllocAlignShell.h"

//================================================================
//
// MallocCore
//
//================================================================

template <typename AddrU>
struct MallocCore
{
    inline stdbool operator()(AddrU& result, AddrU allocSize, stdPars(ErrorLogKit))
    {
        stdBegin;

        REQUIRE(allocSize <= TYPE_MAX(size_t));
        void* allocPtr = malloc(size_t(allocSize));

        require(CHECK_TRACE(allocPtr != 0, CT("Memory allocation failed")));

        COMPILE_ASSERT(sizeof(void*) <= sizeof(AddrU));
        result = (AddrU) allocPtr;

        stdEnd;
    }
};

//================================================================
//
// MallocFlatAllocatorThunk::alloc
//
//================================================================

template <typename AddrU>
bool MallocFlatAllocatorThunk<AddrU>::alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
{
    MallocCore<AddrU> coreAlloc;
    return sysAllocAlignShell<AddrU>(size, alignment, owner, result, coreAlloc, dealloc, stdPassThru);
}

//================================================================
//
// MallocFlatAllocatorThunk::dealloc
//
//================================================================

template <typename AddrU>
void MallocFlatAllocatorThunk<AddrU>::dealloc(MemoryDeallocContext& deallocContext)
{
    AddrU& memAddr = (AddrU&) deallocContext;

    COMPILE_ASSERT(sizeof(void*) == sizeof(size_t));
    requirev(memAddr <= TYPE_MAX(size_t));

    void* memPtr = (void*) memAddr;

    free(memPtr);

    memAddr = 0;
}

//================================================================
//
// Instantiations
//
//================================================================

template class MallocFlatAllocatorThunk<size_t>;
