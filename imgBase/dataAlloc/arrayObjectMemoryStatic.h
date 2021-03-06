#pragma once

#include "storage/uninitializedArray.h"
#include "data/array.h"
#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// ArrayObjectMemoryStatic<T>
//
// The interface is similar to ArrayMemory<T>, but memory is statically reserved to max size.
//
// The constructors/destructors of elements are called on realloc/dealloc.
//
//================================================================

template <typename Type, Space maxSize>
class ArrayObjectMemoryStatic : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = ArrayObjectMemoryStatic<Type, maxSize>;
    COMPILE_ASSERT(maxSize >= 1);

private:

    ArrayObjectMemoryStatic(const SelfType& that); // forbidden
    void operator =(const SelfType& that); // forbidden

public:

    inline ArrayObjectMemoryStatic() {}
    inline ~ArrayObjectMemoryStatic() {dealloc();}

public:

    inline const Array<Type>& operator()() const
        {return *this;}

public:

    inline bool reallocStatic(Space newSize)
    {
        ensure(SpaceU(newSize) <= SpaceU(maxSize));

        dealloc();

        for_count (i, newSize)
            constructDefault(data[i]);

        currentSize = newSize;

        BaseArray::assign(data, newSize, ArrayValidityAssertion{});

        return true;
    }

    inline stdbool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        REQUIRE(reallocStatic(newSize));
        returnTrue;
    }

public:

    void dealloc()
    {
        for (Space i = currentSize-1; i >= 0; --i)
            destruct(data[i]);

        currentSize = 0;

        BaseArray::assignNull();
    }

private:

    // Only elements [0, currentSize) are constructed
    UninitializedArray<Type, maxSize> data;
    Space currentSize = 0;

};

//================================================================
//
// GetSize
//
//================================================================

template <typename Type, Space maxSize>
GET_SIZE_DEFINE(PREP_PASS2(ArrayObjectMemoryStatic<Type, maxSize>), value.size())

//================================================================
//
// ARRAY_OBJECT_STATIC_ALLOC
//
//================================================================

#define ARRAY_OBJECT_STATIC_ALLOC(name, Type, maxSize, size) \
    \
    ArrayObjectMemoryStatic<Type, maxSize> name; \
    require(name.realloc(size, stdPass))
