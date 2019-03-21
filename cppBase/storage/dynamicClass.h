#pragma once

#include "storage/constructDestruct.h"
#include "compileTools/compileTools.h"

//================================================================
//
// DynamicClass
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
// DynamicClass<class MyClassImpl> instance;
//
// In C file, implement MyClassImpl and then implement MyClass functions as thunks,
// using "instance" as a pointer to MyClassImpl instance.
//
//================================================================

template <typename ClassName>
class DynamicClass
{

public:

    inline DynamicClass()
        {memory = new ClassName;}

    inline ~DynamicClass()
        {delete memory;}

    inline ClassName& ref()
        {return *memory;}

    inline const ClassName& ref() const
        {return *memory;}

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

    ClassName* memory;

};
