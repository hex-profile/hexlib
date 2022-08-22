#pragma once

#include "storage/opaqueStruct.h"
#include "numbers/interface/numberInterface.h"
#include "numbers/int/intBase.h"

//================================================================
//
// The interface of resource deallocation 'representative'.
//
// When some resource is allocated, the allocator prepares ResourceOwner instance:
// sets the deallocation function and fills the deallocation context.
//
// When ResourceOwner instance is destroyed, it calls the deallocation function,
// passing the context.
//
// ResourceOwner can be in empty state; in this case, destructor is not called.
//
//================================================================

//================================================================
//
// DeallocContext
//
//================================================================

template <size_t reservedBytes, uint32 hash>
struct DeallocContext : public OpaqueStruct<reservedBytes> {};

////

template <size_t reservedBytes, uint32 hash>
sysinline void exchange(DeallocContext<reservedBytes, hash>& a, DeallocContext<reservedBytes, hash>& b)
{
    OpaqueStruct<reservedBytes>& aBase = a;
    OpaqueStruct<reservedBytes>& bBase = b;
    exchange(aBase, bBase);
}

//================================================================
//
// ResourceOwner
//
// Memory block deallocation representative.
//
//================================================================

template <typename DeallocContext>
class ResourceOwner
{

public:

    typedef void DeallocFunc(DeallocContext& context);

public:

    sysinline ~ResourceOwner()
    {
        if (deallocFunc)
            deallocFunc(context);
    }

    sysinline void clear()
    {
        if (deallocFunc)
            deallocFunc(context);

        deallocFunc = nullptr;
    }

    sysinline DeallocContext& replace(DeallocFunc* newFunc)
    {
        if (deallocFunc)
            deallocFunc(context);

        deallocFunc = newFunc;
        return context;
    }

    sysinline DeallocContext& getContext()
        {return context;}

    sysinline const DeallocContext& getContext() const
        {return context;}

    sysinline void discardAlloc()
        {deallocFunc = nullptr;}

public:

    sysinline friend void exchange(ResourceOwner<DeallocContext>& a, ResourceOwner<DeallocContext>& b)
    {
        exchange(a.context, b.context);
        exchange(a.deallocFunc, b.deallocFunc);
    }

private:

    DeallocContext context;
    DeallocFunc* deallocFunc = nullptr; // 0 in empty state

};
