#pragma once

#include "storage/opaqueStruct.h"
#include "compileTools/compileTools.h"

//================================================================
//
// UninitializedObject
//
//================================================================

template <typename Type>
class UninitializedObject
{

public:

    sysinline Type& operator()()
        {return * (Type*) &rawMemory;}

    sysinline const Type& operator()() const
        {return * (const Type*) &rawMemory;}

private:

    OpaqueStruct<sizeof(Type)> rawMemory;

};

//================================================================
//
// UninitializedArray
//
//================================================================

template <typename Element, size_t nElements>
class UninitializedArray
{

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

    using Array = Element[nElements];
    UninitializedObject<Array> data;

};
