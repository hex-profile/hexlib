#pragma once

#include "dataAlloc/memoryAllocator.h"
#include "errorLog/errorLog.h"

//================================================================
//
// AllocatorInterface
//
//================================================================

template <typename AddrU>
struct AllocatorNull : public AllocatorInterface<AddrU>
{
    virtual stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
    {
        owner.clear();
        result = 0;
        returnTrue;
    }
};

//================================================================
//
// AllocatorForbidden
//
//================================================================

template <typename AddrU>
class AllocatorForbidden : public AllocatorInterface<AddrU>
{

public:

    AllocatorForbidden(const ErrorLogKit& kit)
        : kit{kit} {}

    stdbool alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdNullPars)
    {
        REQUIRE(false);
        returnTrue;
    }

private:

    ErrorLogKit kit;

};
