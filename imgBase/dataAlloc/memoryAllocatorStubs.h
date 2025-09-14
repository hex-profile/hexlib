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
    virtual void alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull)
    {
        owner.clear();
        result = 0;
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

    void alloc(AddrU size, AddrU alignment, MemoryOwner& owner, AddrU& result, stdParsNull)
    {
        REQUIRE(false);
    }

private:

    ErrorLogKit kit;

};
