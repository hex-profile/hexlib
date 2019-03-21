#include "smartPtr.h"

//================================================================
//
// SomeClass
//
//================================================================

class SomeClass
{

public:

    SomeClass(float a, int& b, int&& c) {}

};

//================================================================
//
// compileTestSmartPtr
//
//================================================================

void compileTestSmartPtr(int b, int&& c)
{
    UniquePtr<SomeClass> ptr = makeUnique<SomeClass>(3.14f, b, std::move(c));

    SharedPtr<SomeClass> ptr2 = makeShared<SomeClass>(3.14f, b, std::move(c));
}
