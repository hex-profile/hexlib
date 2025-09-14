#pragma once

#include "errorLog/errorLogKit.h"
#include "stdFunc/stdFunc.h"
#include "dataAlloc/cpuDefaultAlignments.h"
#include "dataAlloc/memoryAllocator.h"
#include "data/array.h"

//================================================================
//
// ArrayMemory<Type>
//
// Allocates/frees array memory.
// Does NOT call element constructors and destructors.
//
// USAGE EXAMPLES:
//
//================================================================

#if 0

// Construct empty array; no memory allocation performed.
ArrayMemory<int> m0;

// Allocate array; check allocation error.
// If reallocation fails, array will have zero size.
m0.realloc(33, stdPass);

// Deallocate memory. Destructor deallocates memory automatically.
m0.dealloc();

// Change array size without reallocation; check error.
// New size should be <= allocated size, otherwise the call fails and size is not changed.
REQUIRE(m0.resize(13));

// Get current allocated size.
REQUIRE(m0.maxSize() == 33);

// Convert to Array<> implicitly and explicitly (for template arguments).
Array<int> tmp0 = m0;
Array<int> tmp1 = m0();

#endif

//================================================================
//
// ArrayMemoryEx<Pointer>
//
// Flexible implementation for any address space.
//
//================================================================

template <typename Pointer>
class ArrayMemoryEx : public ArrayEx<Pointer>
{

    using BaseArray = ArrayEx<Pointer>;

public:

    using AddrU = typename PtrAddrType<Pointer>::AddrU;

public:

    sysinline ArrayMemoryEx()
        {initEmpty();}

    sysinline ~ArrayMemoryEx()
        {dealloc();}

private:

    ArrayMemoryEx(const ArrayMemoryEx<Pointer>& that); // forbidden
    void operator =(const ArrayMemoryEx<Pointer>& that); // forbidden

public:

    sysinline friend void exchange(ArrayMemoryEx<Pointer>& a, ArrayMemoryEx<Pointer>& b)
    {
        BaseArray* ap = &a;
        BaseArray* bp = &b;
        exchange(*ap, *bp);

        exchange(a.memoryDealloc, b.memoryDealloc);
        exchange(a.theAllocPtr, b.theAllocPtr);
        exchange(a.theAllocSize, b.theAllocSize);
    }

public:

    sysinline const ArrayEx<Pointer>& operator()() const
        {return *this;}

    //
    // Reallocation
    //

public:

    void realloc(Space size, Space byteAlignment, AllocatorInterface<AddrU>& allocator, stdPars(ErrorLogKit));

    ////

    void dealloc()
    {
        memoryDealloc.clear();

        theAllocPtr = Pointer(0);
        theAllocSize = 0;

        BaseArray::assignNull(); // clear base array
    }

    ////

    sysinline bool allocated() const {return theAllocPtr != Pointer(0);}

    ////

    sysinline Space maxSize() const {return theAllocSize;}
    sysinline Pointer allocPtr() const {return theAllocPtr;}

    //
    // Resize: rearrange without reallocation
    //

public:

    sysinline void resizeNull()
    {
        BaseArray::assignNull();
    }

    sysinline bool resize(Space size)
    {
        ensure(SpaceU(size) <= SpaceU(theAllocSize));
        BaseArray::assignUnsafe(theAllocPtr, size);

        return true;
    }

private:

    sysinline void initEmpty()
    {
        theAllocPtr = Pointer(0);
        theAllocSize = 0;
    }

private:

    //
    MemoryOwner memoryDealloc;

    // Only for resize
    Pointer theAllocPtr;
    Space theAllocSize;

};

//================================================================
//
// ArrayMemory<Type>
//
// Allocates/frees array memory.
// Does NOT call element constructors and destructors.
//
// C-based address space.
//
//================================================================

template <typename Type>
class ArrayMemory : public ArrayMemoryEx<Type*>
{

    using Base = ArrayMemoryEx<Type*>;

public:

    sysinline friend void exchange(ArrayMemory<Type>& a, ArrayMemory<Type>& b)
    {
        Base* ap = &a;
        Base* bp = &b;
        exchange(*ap, *bp);
    }

public:

    //
    // Default realloc: assumes kit.cpuFastAlloc
    //

    using Base::realloc;

    template <typename Kit>
    sysinline void realloc(Space size, Space byteAlignment, stdPars(Kit))
        {Base::realloc(size, byteAlignment, kit.cpuFastAlloc, stdPassThru);}

    ////

    template <typename Kit>
    sysinline void reallocForCpu(Space size, stdPars(Kit))
        {Base::realloc(size, cpuBaseByteAlignment, kit.cpuFastAlloc, stdPassThru);}

    ////

    template <typename Kit>
    sysinline void reallocForGpuExch(Space size, stdPars(Kit))
        {Base::realloc(size, kit.gpuProperties.samplerAndFastTransferBaseAlignment, kit.cpuFastAlloc, stdPassThru);}

    ////

    template <typename Kit>
    sysinline void reallocInHeap(Space size, stdPars(Kit))
        {Base::realloc(size, maxNaturalAlignment, kit.malloc, stdPassThru);}

    //
    // Cast to Array
    //

    sysinline operator const Array<Type>& () const
    {
        const ArrayEx<Type*>* arr = this;
        return recastEqualLayout<const Array<Type>>(*arr);
    }

    sysinline operator const Array<const Type>& () const
    {
        const ArrayEx<Type*>* arr = this;
        return recastEqualLayout<const Array<const Type>>(*arr);
    }

public:

    //----------------------------------------------------------------
    //
    // operator []
    //
    //----------------------------------------------------------------

    sysinline auto& operator [](ptrdiff_t index) const
    {
        return Base::ptr()[index];
    }

};

//================================================================
//
// ARRAY_ALLOC
// ARRAY_ALLOC_FOR_GPU_EXCH
//
//================================================================

#define ARRAY_ALLOC(name, Type, size) \
    ArrayMemory<Type> name; \
    name.reallocForCpu(size, stdPass);

#define ARRAY_ALLOC_FOR_GPU_EXCH(name, Type, size) \
    ArrayMemory<Type> name; \
    name.reallocForGpuExch(size, stdPass);
