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
inline void exchange(DeallocContext<reservedBytes, hash>& a, DeallocContext<reservedBytes, hash>& b)
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

    inline ResourceOwner()
    {
        deallocFunc = 0;
    }

    inline ~ResourceOwner()
    {
        if (deallocFunc)
            deallocFunc(context);
    }

    inline void clear()
    {
        if (deallocFunc)
            deallocFunc(context);

        deallocFunc = 0;
    }

    inline DeallocContext& replace(DeallocFunc* newFunc)
    {
        if (deallocFunc)
            deallocFunc(context);

        deallocFunc = newFunc;
        return context;
    }

    inline DeallocContext& getContext() {return context;}
    inline const DeallocContext& getContext() const {return context;}

    inline void discardAlloc() {deallocFunc = 0;}

public:

    inline friend void exchange(ResourceOwner<DeallocContext>& a, ResourceOwner<DeallocContext>& b)
    {
        exchange(a.context, b.context);
        exchange(a.deallocFunc, b.deallocFunc);
    }

private:

    DeallocContext context;
    DeallocFunc* deallocFunc; // 0 in empty state

};
