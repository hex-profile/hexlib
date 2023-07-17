#pragma once

#include <vector>

#include "numbers/int/intType.h"
#include "storage/rememberCleanup.h"

//================================================================
//
// GreedyVector
//
// Aggressively preserves capacity.
//
//================================================================

template <typename Type>
class GreedyVector
{

public:

    sysinline GreedyVector() =default;

    sysinline GreedyVector(const GreedyVector& that) =delete;
    sysinline GreedyVector& operator=(const GreedyVector& that) =delete;

    ////

    sysinline void clearMemory()
    {
        count = 0;
        data.clear();
    }

    ////

    sysinline void moveFrom(GreedyVector& that) // no-throw
    {
        data.swap(that.data);
        count = that.count;
        that.count = 0;
    }

    ////

    sysinline auto size() const
        {return count;}

    sysinline void clear()
        {count = 0;}

    ////

    sysinline auto begin() const
        {return data.begin();}

    sysinline auto begin()
        {return data.begin();}

    sysinline auto end() const
        {return data.begin() + count;}

    sysinline auto end()
        {return data.begin() + count;}

    ////

    void appendAtEnd(size_t appendCount) // may throw
    {
        auto reserveAppendCount = minv(data.size() - count, appendCount);

        ////

        auto allocAppendCount = appendCount - reserveAppendCount;

        if (allocAppendCount)
            data.resize(data.size() + allocAppendCount); // atomic for std::vector

        ////

        count += appendCount;
    }

    ////

    void removeFromEnd(size_t n) // no-throw
    {
        n = clampMax(n, count);
        count -= n;
    }

private:

    size_t count = 0; // Always <= size()

    std::vector<Type> data;

};

//================================================================
//
// appendSeq
//
// May throw!
//
//================================================================

template <typename Type, typename Iterator>
void appendSeq(GreedyVector<Type>& buffer, Iterator newPtr, size_t newCount)
{
    buffer.appendAtEnd(newCount);

    REMEMBER_CLEANUP_EX(appendRollback, buffer.removeFromEnd(newCount));

    ////

    auto dstPtr = buffer.end() - newCount;

    for_count (i, newCount)
        *dstPtr++ = std::move(*newPtr++);

    ////

    appendRollback.cancel();
}
