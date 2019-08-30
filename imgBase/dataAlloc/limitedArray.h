#pragma once

#include "data/array.h"
#include "errorLog/errorLog.h"

//================================================================
//
// LimitedArray
//
//================================================================

template <typename Type, Space maxSize>
class LimitedArray : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = LimitedArray<Type, maxSize>;

private:

    sysinline void updateBaseArray()
    {
        BaseArray::assign(&storage[0], currentSize, arrayPreconditionsAreVerified());
    }

public:

    sysinline LimitedArray() =default;

    sysinline SelfType& operator =(const SelfType& that)
    {
        currentSize = that.currentSize;

        for (Space i = 0; i < that.currentSize; ++i)
            storage[i] = that.storage[i];

        updateBaseArray();

        return *this;
    }

public:

    sysinline const Array<Type>& operator()() const
        {return *this;}

public:

    sysinline stdbool resize(Space newSize, stdPars(ErrorLogKit))
    {
        resizeNull();

        REQUIRE(SpaceU(newSize) <= maxSize);
        currentSize = newSize;
        updateBaseArray();

        returnTrue;
    }

    sysinline void resizeNull()
    {
        currentSize = 0;
        updateBaseArray();
    }

public:

    Type* begin() {return &storage[0];}
    const Type* begin() const {return &storage[0];}

    Type* end() {return &storage[currentSize];}
    const Type* end() const {return &storage[currentSize];}

private:

    Space currentSize = maxSize;
    Type storage[maxSize];

};

//================================================================
//
// GetSize
//
//================================================================

template <typename Type, Space maxSize>
GET_SIZE_DEFINE(PREP_PASS2(LimitedArray<Type, maxSize>), value.size())
