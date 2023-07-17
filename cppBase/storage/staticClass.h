#pragma once

#include "storage/opaqueStruct.h"
#include "storage/constructDestruct.h"
#include "compileTools/compileTools.h"

//================================================================
//
// StaticClass
//
//----------------------------------------------------------------
//
// The purpose of the tool is to separate the implementation of a class
// (private data fields and private functions) from its header file.
//
// How to use:
//
// In header file declare a pure class "MyClass", in its private fields write:
//
// StaticClass<class MyClassImpl, nbytes> instance;
//
// In C file, implement MyClassImpl and then implement MyClass functions as thunks,
// using "instance" as a pointer to MyClassImpl instance.
//
//================================================================

template <typename ClassName, size_t classSize>
class StaticClass
{

public:

    inline StaticClass()
    {
        constructDefault(ref());
    }

    inline ~StaticClass()
    {
        destruct(ref());
    }

    inline ClassName& ref()
    {
        return memory.template recast<ClassName>();
    }

    inline const ClassName& ref() const
    {
        return memory.template recast<const ClassName>();
    }

    inline ClassName& operator ~()
        {return ref();}

    inline const ClassName& operator ~() const
        {return ref();}

    inline operator ClassName* ()
        {return &ref();}

    inline operator const ClassName* () const
        {return &ref();}

    inline ClassName* operator ->()
        {return &ref();}

    inline const ClassName* operator ->() const
        {return &ref();}

private:

    OpaqueStruct<classSize, 0> memory;

};
