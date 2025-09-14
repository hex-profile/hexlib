#pragma once

#include <vector>

#include "data/array.h"
#include "errorLog/convertExceptions.h"
#include "errorLog/errorLog.h"
#include "compileTools/blockExceptionsSilent.h"

//================================================================
//
// StlArray
//
//================================================================

template <typename Type>
class StlArray : public Array<Type>
{

public:

    using Element = Type;

private:

    using BaseArray = Array<Type>;
    using SelfType = StlArray<Type>;

private:

    StlArray(const SelfType& that) =delete; // forbidden
    void operator =(const SelfType& that) =delete; // forbidden

public:

    sysinline StlArray() {}
    sysinline ~StlArray() {dealloc();}

public:

    sysinline const Array<Type>& operator()() const
        {return *this;}

public:

    Space maxSize() const
        {return Space(data.size());}

public:

    inline bool resize(Space size)
    {
        ensure(SpaceU(size) <= SpaceU(maxSize()));
        BaseArray::assignUnsafe(size ? &data[0] : nullptr, size);
        return true;
    }

public:

    template <typename Kit>
    sysinline void realloc(Space newSize, stdPars(Kit))
    {
        stdExceptBegin;

        dealloc();

        REQUIRE(newSize >= 0);

        data.resize(newSize);

        REQUIRE(BaseArray::assignValidated(newSize ? &data[0] : nullptr, newSize));

        stdExceptEnd;
    }

public:

    sysinline bool reallocBool(Space newSize)
    {
        boolFuncExceptBegin;

        dealloc();

        ensure(newSize >= 0);
        data.resize(newSize);

        ensure(BaseArray::assignValidated(newSize ? &data[0] : nullptr, newSize));

        boolFuncExceptEnd;
    }

public:

    sysinline void dealloc()
    {
        data.clear();
        BaseArray::assignNull();
    }

private:

    std::vector<Type> data;

};

//================================================================
//
// GetSize
//
//================================================================

template <typename Type>
GET_SIZE_DEFINE(StlArray<Type>, value.size())
