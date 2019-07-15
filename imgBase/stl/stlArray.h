#pragma once

#include <vector>

#include "data/array.h"
#include "errorLog/errorLog.h"

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

    sysinline stdbool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        dealloc();

        REQUIRE(newSize >= 0);

        try
        {
            data.resize(newSize);
        }
        catch (...)
        {
            REQUIRE(false); // memory allocation failed
        }

        BaseArray::assign(newSize ? &data[0] : nullptr, newSize);

        returnTrue;
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
