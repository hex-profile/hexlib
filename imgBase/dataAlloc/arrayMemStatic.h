#pragma once

#include "storage/uninitializedArray.h"
#include "data/array.h"
#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// ArrayMemStatic<T>
//
// The interface is similar to ArrayMemory<T>,
// but memory is statically reserved to max size,
// and constructors/destructors of elements are NOT called.
//
//================================================================

template <typename Type, Space maxSize>
class ArrayMemStatic : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = ArrayMemStatic<Type, maxSize>;
    COMPILE_ASSERT(maxSize >= 1);

private:

    ArrayMemStatic(const SelfType& that); // forbidden
    void operator =(const SelfType& that); // forbidden

public:

    inline ArrayMemStatic() {}
    inline ~ArrayMemStatic() {dealloc();}

public:

    inline const Array<Type>& operator()() const
        {return *this;}

public:

    inline bool reallocStatic(Space newSize)
    {
        ensure(SpaceU(newSize) <= SpaceU(maxSize));

        currentSize = newSize;
        BaseArray::assign(data, newSize, arrayPreconditionsAreVerified());

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
GET_SIZE_DEFINE(PREP_PASS2(ArrayMemStatic<Type, maxSize>), value.size())
