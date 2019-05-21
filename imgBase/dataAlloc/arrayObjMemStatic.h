#pragma once

#include "storage/uninitializedArray.h"
#include "data/array.h"
#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"

//================================================================
//
// ArrayObjMemStatic<T>
//
// The interface is similar to ArrayMemory<T>,
// but memory is statically reserved to max size,
// and constructors/destructors of elements are called.
//
//================================================================

template <typename Type, Space maxSize>
class ArrayObjMemStatic : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = ArrayObjMemStatic<Type, maxSize>;
    COMPILE_ASSERT(maxSize >= 1);

private:

    ArrayObjMemStatic(const SelfType& that); // forbidden
    void operator =(const SelfType& that); // forbidden

public:

    inline ArrayObjMemStatic() {}
    inline ~ArrayObjMemStatic() {dealloc();}

public:

    inline const Array<Type>& operator()() const
        {return *this;}

public:

    inline bool reallocStatic(Space newSize)
    {
        ensure(SpaceU(newSize) <= SpaceU(maxSize));

        dealloc();

        for (Space i = 0; i < newSize; ++i)
            constructDefault(data[i]);

        currentSize = newSize;

        BaseArray::assign(data, newSize, arrayPreconditionsAreVerified());

        return true;
    }

    inline stdbool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        stdBegin;
        REQUIRE(reallocStatic(newSize));
        stdEnd;
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
