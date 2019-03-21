#pragma once

#include "errorLog/errorLog.h"
#include "data/space.h"
#include "allocation/flatMemoryAllocator.h"

//================================================================
//
// FlatMemoryHolder
//
//================================================================

template <typename AddrU>
class FlatMemoryHolder
{

public:

    inline FlatMemoryHolder()
    {
        allocPtr = 0;
        allocSize = 0;
        currentSize = 0;
    }

    inline ~FlatMemoryHolder()
        {dealloc();}

private:

    FlatMemoryHolder(const FlatMemoryHolder& that); // forbidden
    void operator =(const FlatMemoryHolder& that); // forbidden

public:

    inline bool realloc(AddrU size, AddrU alignment, FlatMemoryAllocator<AddrU>& allocator, stdNullPars)
    {
        stdNullBegin;

        AddrU newPtr = 0;
        require(allocator.alloc(size, alignment, memoryOwner, newPtr, stdNullPassThru));

        allocPtr = newPtr;
        allocSize = size;

        currentSize = size;

        stdEnd;
    }

    ////

    inline void dealloc()
    {
        memoryOwner.clear();

        allocPtr = 0;
        allocSize = 0;

        currentSize = 0;
    }

    ////

    inline AddrU maxSize() const {return allocSize;}

    ////

    inline AddrU ptr() const {return allocPtr;}
    inline AddrU size() const {return currentSize;}

    ////

    inline void resizeNull()
        {currentSize = 0;}

    bool resize(AddrU size) // rearrange without reallocation
    {
        require(SpaceU(size) <= SpaceU(allocSize));
        currentSize = size;

        return true;
    }

private:

    MemoryOwner memoryOwner;

    AddrU allocPtr;
    AddrU allocSize;

    AddrU currentSize;

};
