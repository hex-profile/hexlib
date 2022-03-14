#pragma once

#include "storage/uninitializedArray.h"
#include "dbgptr/dbgptrGate.h"
#include "storage/constructDestruct.h"
#include "errorLog/errorLog.h"
#include "stdFunc/stdFunc.h"
#include "data/array.h"

//================================================================
//
// ArrayObjectMemoryStatic<T>
//
// The interface is similar to ArrayMemory<T>, but memory is statically reserved to max size.
//
// The constructors/destructors of elements are called on realloc/dealloc.
//
//================================================================

template <typename TypeParam, Space maxSize>
class ArrayObjectMemoryStatic
{

    //----------------------------------------------------------------
    //
    // Defs.
    //
    //----------------------------------------------------------------

public:

    using Type = TypeParam;

    COMPILE_ASSERT(maxSize >= 1);

    //----------------------------------------------------------------
    //
    // Construct / destruct.
    //
    //----------------------------------------------------------------

public:

    sysinline ArrayObjectMemoryStatic() 
        {}

    sysinline ~ArrayObjectMemoryStatic() 
        {dealloc();}

    //----------------------------------------------------------------
    //
    // Copy-construct and assign.
    //
    //----------------------------------------------------------------

public:

    template <typename T>
    explicit ArrayObjectMemoryStatic(const ArrayObjectMemoryStatic<T, maxSize>& that)
    {
        for_count (i, that.currentSize)
            constructCopy(data[i], that.data[i]);

        currentSize = that.currentSize;
    }

public:

    template <typename T>
    auto& operator =(const ArrayObjectMemoryStatic<T, maxSize>& that)
    {
        dealloc();

        for_count (i, that.currentSize)
            constructCopy(data[i], that.data[i]);

        currentSize = that.currentSize;

        return *this;
    }

    //----------------------------------------------------------------
    //
    // Get array.
    //
    //----------------------------------------------------------------

private:

    sysinline Array<Type> getArray()
        {return {data(), currentSize, ArrayValidityAssertion{}};}

    sysinline Array<const Type> getArray() const
        {return {data(), currentSize, ArrayValidityAssertion{}};}

public:

    sysinline auto operator()()
        {return getArray();}

    sysinline auto operator()() const
        {return getArray();}

    sysinline operator Array<Type> ()
        {return getArray();}

    sysinline operator Array<const Type> () const 
        {return getArray();}

    //----------------------------------------------------------------
    //
    // Realloc and dealloc.
    //
    //----------------------------------------------------------------

public:

    sysinline bool reallocStatic(Space newSize)
    {
        ensure(SpaceU(newSize) <= SpaceU(maxSize));

        dealloc();

        for_count (i, newSize)
            constructDefault(data[i]);

        currentSize = newSize;

        return true;
    }

public:

    sysinline stdbool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        REQUIRE(reallocStatic(newSize));
        returnTrue;
    }

public:

    void dealloc()
    {
        for (Space i = currentSize-1; i >= 0; --i)
            destruct(data[i]);

        currentSize = 0;
    }

    //----------------------------------------------------------------
    //
    // API for ptr() and size()
    //
    //----------------------------------------------------------------

public:

    sysinline Type* ptrUnsafeForInternalUseOnly()
        {return data();}

    sysinline const Type* ptrUnsafeForInternalUseOnly() const
        {return data();}

#if HEXLIB_GUARDED_MEMORY

    sysinline auto ptr()
        {return ArrayPtrCreate(Type, data(), currentSize, DbgptrArrayPreconditions());}

    sysinline auto ptr() const
        {return ArrayPtrCreate(const Type, data(), currentSize, DbgptrArrayPreconditions());}

#else

    sysinline Type* ptr()
        {return data();}

    sysinline const Type* ptr() const
        {return data();}

#endif

    sysinline Space size() const // always >= 0
        {return currentSize;}

    //----------------------------------------------------------------
    //
    // operator []
    //
    //----------------------------------------------------------------

public:

    sysinline bool validAccess(Space index) const
        {return SpaceU(index) < SpaceU(currentSize);}

public:

    sysinline Type& operator [](Space index)
        {return ptr()[index];}

    sysinline const Type& operator [](Space index) const
        {return ptr()[index];}

    //----------------------------------------------------------------
    //
    // State.
    //
    //----------------------------------------------------------------

private:

    // Only elements [0, currentSize) are constructed
    UninitializedArray<Type, maxSize> data;
    Space currentSize = 0;

};

//================================================================
//
// GetSize
//
//================================================================

template <typename Type, Space maxSize>
GET_SIZE_DEFINE(PREP_PASS2(ArrayObjectMemoryStatic<Type, maxSize>), value.size())

//================================================================
//
// ARRAY_OBJECT_STATIC_ALLOC
//
//================================================================

#define ARRAY_OBJECT_STATIC_ALLOC(name, Type, maxSize, size) \
    \
    ArrayObjectMemoryStatic<Type, maxSize> name; \
    require(name.realloc(size, stdPass))
