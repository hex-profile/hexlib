#pragma once

#include "stdFunc/stdFunc.h"
#include "gpuLayer/gpuLayer.h"
#include "errorLog/debugBreak.h"
#include "storage/nonCopyable.h"

//================================================================
//
// GpuBufferState
//
//================================================================

struct GpuBufferState
{
    GpuMemoryOwner allocOwner;
    GpuAddrU allocPtr = 0;
    GpuAddrU allocSize = 0;
    GpuAddrU allocAlignment = 1;
};

//================================================================
//
// GpuBuffer
//
//================================================================

class GpuBuffer : private GpuBufferState
{

public:

    GpuBuffer() =default;

    GpuBuffer(const GpuBuffer& that) =delete;
    GpuBuffer& operator =(const GpuBuffer& that) =delete;

    GpuBuffer(GpuBuffer&& that)
    {
        GpuBufferState& thisState = *this;
        GpuBufferState& thatState = that;

        thisState = thatState;
        thatState = {};
    }

    GpuBuffer& operator =(GpuBuffer&& that) =delete;

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

    template <typename Type, typename Kit>
    void getArray(GpuArray<Type>& result, stdPars(Kit)) const
    {
        REQUIRE(size() <= spaceMax);

        auto bytes = SpaceU(size());
        auto count = bytes / SpaceU(sizeof(Type));
        REQUIRE(count * SpaceU(sizeof(Type)) == bytes);

        result.assignUnsafe(ptr<Type>(), Space(count));
    }

public:

    template <typename Kit>
    void realloc(size_t size, size_t alignment, stdPars(Kit))
    {
        dealloc();

        /////

        auto& allocator = kit.gpuMemoryAllocation.gpuAllocator();

        ////

        GpuAddrU ptr{};
        allocator.alloc(kit.gpuCurrentContext, size, alignment, allocOwner, ptr, stdPass);

        allocPtr = ptr;
        allocSize = size;
        allocAlignment = alignment;
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

};
