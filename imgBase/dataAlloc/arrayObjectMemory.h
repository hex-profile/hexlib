#pragma once

#include "dataAlloc/arrayMemory.h"
#include "storage/constructDestruct.h"

//================================================================
//
// The same as ArrayMemory<T> but with support of element constructors/destructors.
//
// The constructors/destructors of elements are called only on realloc/dealloc.
// Resize functions do not call constructors or destructors.
//
//================================================================

#if 0

// Construct empty array; no memory allocation performed.
ArrayObjectMemory<MyClass> m0;

// Allocate array, call default constructors of elements and check error.
// If reallocation fails, array has zero size.
REQUIRE(m0.realloc(33, stdPass));

// Call element destructors and deallocate memory.
// The container's destructor will call element destructors and deallocate memory automatically.
m0.dealloc();

// Change array size without reallocation; check error.
// New size should be <= allocated size, otherwise the call fails and size is not changed.
// No constructors/destructors of elements are called.
REQUIRE(m0.resize(13));

// Get current allocated size.
REQUIRE(m0.maxSize() == 33);

#endif

//================================================================
//
// ArrayObjectMemory<T>
//
//================================================================

template <typename Type>
class ArrayObjectMemory : public ArrayMemoryEx<Type*>
{

    using Base = ArrayMemoryEx<Type*>;
    using AddrU = typename Base::AddrU;

public:

    using Element = Type;

public:

    //
    // Override to call redefined realloc/dealloc
    //

    inline ArrayObjectMemory()
        {}

    inline ~ArrayObjectMemory()
        {dealloc();}

    //
    // realloc
    //

    stdbool reallocEx(Space size, Space byteAlignment, AllocatorInterface<AddrU>& allocator, bool dataProcessing, stdPars(ErrorLogKit));

    ////

    template <typename Kit>
    inline stdbool reallocInHeap(Space size, stdPars(Kit))
        {return reallocEx(size, maxNaturalAlignment, kit.malloc, true, stdPassThru);}

    //
    // dealloc
    //

    void dealloc();

private:

    bool constructed = false;

};

//================================================================
//
// ARRAY_OBJECT_HEAP_ALLOC
//
//================================================================

#define ARRAY_OBJECT_HEAP_ALLOC(name, Type, size) \
    \
    ArrayObjectMemory<Type> name; \
    require(name.reallocInHeap(size, stdPass));
