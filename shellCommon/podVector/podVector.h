#pragma once

#include "storage/nonCopyable.h"
#include "numbers/int/intType.h"
#include "compileTools/gccBugfix.h"

//================================================================
//
// PodVector
//
// Like std::vector, but it doesn't call constructors and destructors.
//
// Some methods may throw exceptions.
// Highly efficient.
//
//================================================================

class PodVectorBase : public NonCopyable
{

protected:

    sysinline PodVectorBase()
        {}

    sysinline ~PodVectorBase()
        {deallocBytes();}

    ////

    [[noreturn]]
    void throwError();

    ////

    void deallocBytes();

    void reallocToBytes(size_t newSizeInBytes) may_throw;

    bool tryReallocToBytes(size_t newSizeInBytes);

    ////

    void* allocPtr = nullptr;

    ////

    sysinline void swap(PodVectorBase& that)
    {
        exchange(allocPtr, that.allocPtr);
    }

};

//================================================================
//
// PodVector
//
//================================================================

template <typename Type>
class PodVector : private PodVectorBase
{

public:

    //----------------------------------------------------------------
    //
    // Init & deinit.
    //
    //----------------------------------------------------------------

    sysinline PodVector()
        {}

    sysinline ~PodVector()
        {}

    sysinline PodVector(size_t size, bool amortized) may_throw
        {resize(size, amortized);}

    //----------------------------------------------------------------
    //
    // Dealloc.
    //
    //----------------------------------------------------------------

    sysinline void dealloc()
    {
        deallocBytes();
        currentSize = 0;
        allocSize = 0;
    }

    //----------------------------------------------------------------
    //
    // Swap.
    //
    //----------------------------------------------------------------

    sysinline void swap(PodVector<Type>& that)
    {
        PodVectorBase::swap(that);
        exchange(currentSize, that.currentSize);
        exchange(allocSize, that.allocSize);
    }

    //----------------------------------------------------------------
    //
    // Accessors.
    //
    //----------------------------------------------------------------

    sysinline size_t size() const
        {return currentSize;}

    sysinline bool empty() const
        {return currentSize == 0;}

    sysinline size_t capacity() const
        {return allocSize;}

    ////

    sysinline auto data()
        {return (Type*) allocPtr;}

    sysinline auto data() const
        {return (const Type*) allocPtr;}

    ////

    sysinline Type& operator[](size_t i)
        {return ((Type*) allocPtr)[i];}

    sysinline const Type& operator[](size_t i) const
        {return ((const Type*) allocPtr)[i];}

    //----------------------------------------------------------------
    //
    // "Iterators"
    //
    //----------------------------------------------------------------

    sysinline auto* begin()
        {return data();}

    sysinline auto* begin() const
        {return data();}

    sysinline auto* end()
        {return data() + currentSize;}

    sysinline auto* end() const
        {return data() + currentSize;}

    //----------------------------------------------------------------
    //
    // clear
    //
    //----------------------------------------------------------------

public:

    sysinline void clear()
        {currentSize = 0;}

    //----------------------------------------------------------------
    //
    // back
    // push_back
    // pop_back
    //
    //----------------------------------------------------------------

public:

    sysinline Type& back()
        {return (*this)[currentSize-1];}

    sysinline const Type& back() const
        {return (*this)[currentSize-1];}

public:

    sysinline void push_back(const Type& value) may_throw
    {
        if (currentSize < allocSize)
        {
            (*this)[currentSize] = value;
            currentSize++;
            return;
        }

        ////

        auto oldSize = currentSize;

        if_not (currentSize <= maxSize - 1)
            throwError();

        resizeImpl(currentSize + 1, true);

        (*this)[oldSize] = value;
    }

public:

    sysinline void pop_back()
    {
        if (currentSize)
            currentSize--;
    }

    //----------------------------------------------------------------
    //
    // reserve
    //
    //----------------------------------------------------------------

public:

    sysinline void reserve(size_t newSize) may_throw
    {
        if (newSize <= allocSize)
            return;

        reallocExact(newSize);
    }

    //----------------------------------------------------------------
    //
    // resize
    //
    //----------------------------------------------------------------

public:

    sysinline void resize(size_t newSize, bool amortized) may_throw
    {
        if (newSize <= allocSize)
        {
            currentSize = newSize;
            return;
        }

        resizeImpl(newSize, amortized);
    }

    //----------------------------------------------------------------
    //
    // shrink_to_fit
    //
    //----------------------------------------------------------------

public:

    void shrink_to_fit()
    {
        if (currentSize != allocSize)
        {
            if (tryReallocToBytes(currentSize * sizeof(Type)))
                allocSize = currentSize;
        }
    }

    //----------------------------------------------------------------
    //
    // append_uninit
    //
    //----------------------------------------------------------------

public:

    sysinline void append_uninit(size_t size, bool amortized) may_throw
    {
        if (size == 0)
            return;

        ////

        if_not (size <= maxSize && currentSize <= maxSize - size)
            throwError();

        resize(currentSize + size, amortized);
    }

    //----------------------------------------------------------------
    //
    // append
    //
    //----------------------------------------------------------------

public:

    sysinline void append(const Type* ptr, size_t size, bool amortized) may_throw
    {
        auto oldSize = currentSize;

        ////

        append_uninit(size, amortized);

        ////

        auto dstPtr = data() + oldSize;

        if (size <= 8)
        {
            for_count (i, size)
                *dstPtr++ = *ptr++;
        }
        else
        {
            GCC_BUGFIX_PRAGMA("GCC diagnostic push")
            GCC_BUGFIX_PRAGMA("GCC diagnostic ignored \"-Warray-bounds\"")

            memcpy(dstPtr, ptr, size * sizeof(Type));

            GCC_BUGFIX_PRAGMA("GCC diagnostic pop")
        }
    }

    //----------------------------------------------------------------
    //
    // append_value
    //
    //----------------------------------------------------------------

public:

    sysinline void append_value(const Type& value, size_t size, bool amortized) may_throw
    {
        auto oldSize = currentSize;

        ////

        append_uninit(size, amortized);

        ////

        auto dstPtr = data() + oldSize;

        for_count (i, size)
            *dstPtr++ = value;
    }

    //----------------------------------------------------------------
    //
    // assign
    //
    //----------------------------------------------------------------

public:

    sysinline void assign(const Type* ptr, size_t size, bool amortized) may_throw
    {
        clear();
        append(ptr, size, amortized);
    }

    //----------------------------------------------------------------
    //
    // reallocExact.
    //
    // Reallocates exactly to the size.
    //
    //----------------------------------------------------------------

private:

    sysinline void reallocExact(size_t newSize) may_throw
    {
        if_not (newSize <= maxSize)
            throwError();

        reallocToBytes(newSize * sizeof(Type));

        allocSize = newSize;
    }

    //----------------------------------------------------------------
    //
    // reallocAmortized.
    //
    // Reallocates with exponential growth.
    //
    //----------------------------------------------------------------

private:

    sysinline void reallocAmortized(size_t newSize) may_throw
    {
        if_not (newSize <= maxSize)
            throwError();

        ////

        auto desiredSize = newSize;

        auto extraSize = (newSize >> 2); // Factor 1.25X

        if (desiredSize <= maxSize - extraSize)
            desiredSize += extraSize;

        ////

        if (tryReallocToBytes(desiredSize * sizeof(Type)))
        {
            allocSize = desiredSize;
            return;
        }

        reallocToBytes(newSize * sizeof(Type));
        allocSize = newSize;
    }

    //----------------------------------------------------------------
    //
    // resizeImpl
    //
    //----------------------------------------------------------------

private:

    sysinline void resizeImpl(size_t newSize, bool amortized) may_throw
    {
        if (amortized)
            reallocAmortized(newSize);
        else
            reallocExact(newSize);

        currentSize = newSize;
    }

    //----------------------------------------------------------------
    //
    // Data.
    //
    //----------------------------------------------------------------

private:

    static constexpr size_t maxSize = TYPE_MAX(size_t) / sizeof(Type);

    size_t currentSize = 0; // <= allocSize
    size_t allocSize = 0;

};
