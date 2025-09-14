#pragma once

#include "errorLog/errorLog.h"
#include "dataAlloc/memoryAllocator.h"

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
    {
        dealloc();
    }

private:

    FlatMemoryHolder(const FlatMemoryHolder& that); // forbidden
    void operator =(const FlatMemoryHolder& that); // forbidden

public:

    inline void realloc(AddrU size, AddrU alignment, AllocatorInterface<AddrU>& allocator, stdParsNull)
    {
        AddrU newPtr = 0;
        allocator.alloc(size, alignment, memoryOwner, newPtr, stdPassNullThru);

        allocPtr = newPtr;
        allocSize = size;

        currentSize = size;
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
    {
        currentSize = 0;
    }

    bool resize(AddrU size) // rearrange without reallocation
    {
        currentSize = size;

        return true;
    }

private:

    MemoryOwner memoryOwner;

    AddrU allocPtr;
    AddrU allocSize;

    AddrU currentSize;

};
