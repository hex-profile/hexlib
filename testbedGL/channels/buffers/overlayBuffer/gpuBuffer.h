#pragma once

#include "stdFunc/stdFunc.h"
#include "gpuLayer/gpuLayer.h"
#include "errorLog/debugBreak.h"
#include "storage/nonCopyable.h"

//================================================================
//
// GpuBuffer
//
//================================================================

class GpuBuffer : public NonCopyable
{

public:

    template <typename Type>
    auto ptr() const
    {
        using Ptr = GpuPtr(Type);
        return Ptr(allocPtr);
    }

    auto size() const
    {
        return allocSize;
    }

    auto alignment() const
    {
        return allocAlignment;
    }

public:

    template <typename Kit>
    stdbool realloc(size_t size, size_t alignment, stdPars(Kit))
    {
        dealloc();

        /////

        auto& allocator = kit.gpuMemoryAllocation.gpuAllocator();

        ////

        GpuAddrU ptr{};
        require(allocator.alloc(kit.gpuCurrentContext, size, alignment, allocOwner, ptr, stdPass));

        allocPtr = ptr;
        allocSize = size;
        allocAlignment = alignment;

        returnTrue;
    }

    void dealloc()
    {
        allocOwner.clear();
        allocPtr = 0;
        allocSize = 0;
        allocAlignment = 1;
    }

public:

    friend inline void exchange(GpuBuffer& a, GpuBuffer& b)
    {
        exchange(a.allocOwner, b.allocOwner);
        exchange(a.allocPtr, b.allocPtr);
        exchange(a.allocSize, b.allocSize);
        exchange(a.allocAlignment, b.allocAlignment);
    }

private:

    GpuMemoryOwner allocOwner;
    GpuAddrU allocPtr = 0;
    GpuAddrU allocSize = 0;
    GpuAddrU allocAlignment = 1;

};
