#include "mallocAllocator.h"

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
        REQUIRE(allocSize <= TYPE_MAX(size_t));
        void* allocPtr = malloc(size_t(allocSize));

        REQUIRE_CUSTOM(allocPtr != 0, CT("Memory allocation failed"));

        COMPILE_ASSERT(sizeof(void*) <= sizeof(AddrU));
        result = (AddrU) allocPtr;

        returnTrue;
    }
};

//================================================================
//
// MallocAllocator::alloc
//
//================================================================

template <typename AddrU>
stdbool MallocAllocator<AddrU>::alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
{
    MallocCore<AddrU> coreAlloc;
    return sysAllocAlignShell<AddrU>(size, alignment, owner, result, coreAlloc, dealloc, stdPassThru);
}

//================================================================
//
// MallocAllocator::dealloc
//
//================================================================

template <typename AddrU>
void MallocAllocator<AddrU>::dealloc(MemoryDeallocContext& deallocContext)
{
    AddrU& memAddr = (AddrU&) deallocContext;

    COMPILE_ASSERT(sizeof(void*) == sizeof(size_t));
    ensurev(memAddr <= TYPE_MAX(size_t));

    void* memPtr = (void*) memAddr;

    free(memPtr);

    memAddr = 0;
}

//================================================================
//
// Instantiations
//
//================================================================

template class MallocAllocator<size_t>;
