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

    sysinline bool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        stdBegin;

        dealloc();

        try
        {
            REQUIRE(newSize >= 0);
            data.resize(newSize);
        }
        catch (...)
        {
            REQUIRE(false); // memory allocation failed
        }

        BaseArray::assign(newSize ? &data[0] : nullptr, newSize);

        stdEnd;
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