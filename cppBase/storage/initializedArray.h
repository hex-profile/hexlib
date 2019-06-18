#pragma once

#include "storage/uninitializedArray.h"

//================================================================
//
// InitializedArray
//
//================================================================

template <typename Type, size_t size>
class InitializedArray
{

public:

    using Element = Type;

private:

    using SelfType = InitializedArray<Type, size>;
    COMPILE_ASSERT(size >= 1);

public:

    sysinline operator Element* ()
        {return data();}

    sysinline operator const Element* () const
        {return data();}

    sysinline Element* operator()()
        {return data();}

    sysinline const Element* operator()() const
        {return data();}

private:

    InitializedArray(const SelfType& that) =delete;
    void operator =(const SelfType& that) =delete;

public:

    sysinline InitializedArray()
    {
        for (size_t i = 0; i < size; ++i)
            constructDefaultInit(data[i]);
    }

    sysinline InitializedArray(const Type& value)
    {
        for (size_t i = 0; i < size; ++i)
            constructCopy(data[i], value);
    }

    sysinline ~InitializedArray() 
    {
        for (size_t i = size; i >= 1; --i)
            destruct(data[i-1]);
    }

private:

    UninitializedArray<Type, size> data;

};
