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

private:

    template <typename T>
    sysinline void copyConstruct(const ArrayObjectMemoryStatic<T, maxSize>& that)
    {
        for_count (i, that.allocSize)
            constructCopy(data[i], that.data[i]);

        allocSize = that.allocSize;
        usedSize = that.usedSize;
    }

public:

    template <typename T>
    sysinline explicit ArrayObjectMemoryStatic(const ArrayObjectMemoryStatic<T, maxSize>& that)
    {
        copyConstruct(that);
    }

    sysinline explicit ArrayObjectMemoryStatic(const ArrayObjectMemoryStatic<Type, maxSize>& that)
    {
        copyConstruct(that);
    }

public:

    template <typename T>
    sysinline auto& operator =(const ArrayObjectMemoryStatic<T, maxSize>& that)
    {
        if_not (this == &that)
            {dealloc(); copyConstruct(that);}

        return *this;
    }

    sysinline auto& operator =(const ArrayObjectMemoryStatic<Type, maxSize>& that)
    {
        if_not (this == &that)
            {dealloc(); copyConstruct(that);}

        return *this;
    }

    //----------------------------------------------------------------
    //
    // Get array.
    //
    //----------------------------------------------------------------

private:

    sysinline Array<Type> getArray()
        {return {data(), usedSize, ArrayValidityAssertion{}};}

    sysinline Array<const Type> getArray() const
        {return {data(), usedSize, ArrayValidityAssertion{}};}

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

        allocSize = newSize;
        usedSize = newSize;

        return true;
    }

public:

    sysinline stdbool realloc(Space newSize, stdPars(ErrorLogKit))
    {
        REQUIRE(reallocStatic(newSize));
        returnTrue;
    }

public:

    sysinline void dealloc()
    {
        for (auto i = allocSize - 1; i >= 0; --i)
            destruct(data[i]);

        allocSize = 0;
        usedSize = 0;
    }

    //----------------------------------------------------------------
    //
    // Resize.
    //
    //----------------------------------------------------------------

public:

    sysinline void resizeNull() 
    {
        usedSize = 0;
    }

    sysinline bool resize(Space newSize)
    {
        ensure(0 <= newSize && newSize <= allocSize);
        usedSize = newSize;
        return true;
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
        {return ArrayPtrCreate(Type, data(), usedSize, DbgptrArrayPreconditions());}

    sysinline auto ptr() const
        {return ArrayPtrCreate(const Type, data(), usedSize, DbgptrArrayPreconditions());}

#else

    sysinline Type* ptr()
        {return data();}

    sysinline const Type* ptr() const
        {return data();}

#endif

    sysinline Space size() const // always >= 0
        {return usedSize;}

    //----------------------------------------------------------------
    //
    // operator []
    //
    //----------------------------------------------------------------

public:

    sysinline bool validAccess(Space index) const
        {return SpaceU(index) < SpaceU(usedSize);}

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

    // Only elements [0, allocSize) are constructed
    UninitializedArray<Type, maxSize> data;

    Space allocSize = 0; // [0, maxSize]
    Space usedSize = 0; // [0, allocSize]

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
