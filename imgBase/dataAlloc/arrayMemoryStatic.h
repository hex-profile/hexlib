#pragma once

#include "data/array.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "storage/constructDestruct.h"
#include "storage/uninitializedArray.h"

//================================================================
//
// ArrayMemoryStatic<T>
//
// The interface is similar to ArrayMemory<T>,
// but memory is statically reserved to max size,
// and constructors/destructors of elements are NOT called.
//
//================================================================

template <typename Type, Space maxSize>
class ArrayMemoryStatic : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = ArrayMemoryStatic<Type, maxSize>;
    COMPILE_ASSERT(maxSize >= 1);

private:

    ArrayMemoryStatic(const SelfType& that); // forbidden
    void operator =(const SelfType& that); // forbidden

public:

    sysinline ArrayMemoryStatic() {}
    sysinline ~ArrayMemoryStatic() {dealloc();}

public:

    sysinline operator const Array<const Type>& () const
    {
        const BaseArray* base = this;
        return recastEqualLayout<const Array<const Type>>(*base);
    }

public:

    sysinline bool reallocStatic(Space newSize)
    {
        ensure(SpaceU(newSize) <= SpaceU(maxSize));

        currentSize = newSize;
        BaseArray::assignUnsafe(data(), newSize);

        return true;
    }

    sysinline void realloc(Space newSize, stdPars(ErrorLogKit))
    {
        REQUIRE(reallocStatic(newSize));
    }

public:

    void dealloc()
    {
        currentSize = 0;
        BaseArray::assignNull();
    }

private:

    UninitializedArray<Type, maxSize> data;
    Space currentSize = 0;

};

//================================================================
//
// GetSize
//
//================================================================

template <typename Type, Space maxSize>
GET_SIZE_DEFINE(PREP_PASS2(ArrayMemoryStatic<Type, maxSize>), value.size())

//================================================================
//
// ARRAY_STATIC_ALLOC
//
//================================================================

#define ARRAY_STATIC_ALLOC(name, Type, maxSize, size) \
    \
    ArrayMemoryStatic<Type, maxSize> name; \
    name.realloc(size, stdPass);
