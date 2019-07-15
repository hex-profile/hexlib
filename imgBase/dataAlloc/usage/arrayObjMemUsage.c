#include "dataAlloc/arrayObjMem.inl"
#include "errorLog/errorLog.h"

//================================================================
//
// MyClass
//
//================================================================

class MyClass
{

public:

    MyClass() {ptr = malloc(1);}
    ~MyClass() {free(ptr);}

private:

    MyClass(const MyClass& that); // forbidden
    void operator= (const MyClass& that); // forbidden;

private:

    void* ptr;

};

//================================================================
//
// TestKit
//
//================================================================

KIT_COMBINE3(TestKit, CpuFastAllocKit, ErrorLogKit, DataProcessingKit);

//================================================================
//
// arrayObjMemUsage
//
//================================================================

stdbool arrayObjMemUsage(stdPars(TestKit))
{
    // Construct empty array; no memory allocation performed.
    ArrayObjMem<MyClass> m0;

    // Allocate array, call default constructors of elements and check error.
    // If reallocation fails, array has zero size.
    require(m0.realloc(33, stdPass));

    // Call element destructors and deallocate memory.
    // The container's destructor will call element destructors and deallocate memory automatically.
    m0.dealloc();

    // Change array size without reallocation; check error.
    // New size should be <= allocated size, otherwise the call fails and size is not changed.
    // No constructors/destructors of elements are called.
    REQUIRE(m0.resize(13));

    // Get current allocated size.
    REQUIRE(m0.maxSize() == 33);

    returnTrue;
}
