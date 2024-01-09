#pragma once

#include "data/commonFuncs.h"
#include "data/pointerInterface.h"
#include "dbgptr/dbgptrGate.h"
#include "errorLog/debugBreak.h"
#include "extLib/data/arrayBase.h"
#include "numbers/int/intType.h"

//================================================================
//
// Array<Type>
//
// Array memory layout description: ptr and size.
//
// ptr:
// Points to 0th array element. Can be NULL if the array is empty.
//
// size:
// The array size. Always >= 0.
// If size is zero, the array is empty.
//
//================================================================

//================================================================
//
// ARRAY__CHECK_POINTER
//
//================================================================

template <typename SrcPointer, typename DstPointer>
sysinline auto arrayCheckPointerConversion()
{
    SrcPointer srcPtr(0);
    DstPointer dstPtr = srcPtr;
    return dstPtr;
}

//----------------------------------------------------------------

#define ARRAY__CHECK_POINTER(SrcPointer, DstPointer) \
    COMPILE_ASSERT(sizeof(arrayCheckPointerConversion<SrcPointer, DstPointer>()) >= 1)

//================================================================
//
// ARRAY_EXPOSE
//
//================================================================

#define ARRAY_EXPOSE_EX2(array, arrayPtr, arraySize) \
    auto arrayPtr = (array).ptr(); \
    auto arraySize = (array).size();

#define ARRAY_EXPOSE_EX(array, prefix) \
    ARRAY_EXPOSE_EX2(array, prefix##Ptr, prefix##Size)

#define ARRAY_EXPOSE(array) \
    ARRAY_EXPOSE_EX(array, array)

//----------------------------------------------------------------

#define ARRAY_EXPOSE_UNSAFE_EX2(array, arrayPtr, arraySize) \
    auto arrayPtr = (array).ptrUnsafeForInternalUseOnly(); \
    auto arraySize = (array).size()

#define ARRAY_EXPOSE_UNSAFE_EX(array, prefix) \
    ARRAY_EXPOSE_UNSAFE_EX2(array, prefix##Ptr, prefix##Size)

#define ARRAY_EXPOSE_UNSAFE(array) \
    ARRAY_EXPOSE_UNSAFE_EX(array, array)

//================================================================
//
// ARRAY_VALID_ACCESS
//
//================================================================

#define ARRAY_VALID_ACCESS(array, pos) \
    (SpaceU(pos) < SpaceU(array##Size))

sysinline bool arrayValidAccess(Space size, Space pos)
    {return SpaceU(pos) < SpaceU(size);}

//================================================================
//
// ArrayEx<Pointer>
//
// Supports custom address space.
//
//================================================================

template <typename Pointer>
class ArrayEx
    :
    public ArrayBase<typename PtrElemType<Pointer>::T, Pointer>
{

    //----------------------------------------------------------------
    //
    // Types.
    //
    //----------------------------------------------------------------

public:

    using Type = typename PtrElemType<Pointer>::T;

private:

    using Base = ArrayBase<Type, Pointer>;
    using Base::thePtr;
    using Base::theSize;

    ////

    template <typename OtherPointer>
    friend class ArrayEx;

public:

    //----------------------------------------------------------------
    //
    // Construct.
    //
    //----------------------------------------------------------------

    sysinline ArrayEx() {}

    //----------------------------------------------------------------
    //
    // Assign.
    //
    //----------------------------------------------------------------

    using Base::assignValidated;
    using Base::assignNull;

    ////

    template <typename OtherPointer>
    sysinline void assignUnsafe(OtherPointer ptr, Space size)
    {
        if_not (arrayBaseIsValid<sizeof(Type)>(size)) // quick check
            {DEBUG_BREAK_INLINE(); size = 0;}

        thePtr = ptr;
        theSize = size;
    }

    //----------------------------------------------------------------
    //
    // Export cast (no code generated, reinterpret 'this')
    //
    //----------------------------------------------------------------

    using ConstPointer = typename PtrRebaseType<Pointer, const Type>::T;
    struct DummyType;
    using ExportType = TypeSelect<TYPE_EQUAL(Type, const Type), DummyType, ArrayEx<ConstPointer>>;

    ////

    sysinline operator const ExportType& () const
    {
        return recastEqualLayout<const ExportType>(*this);
    }

    //----------------------------------------------------------------
    //
    // Get size
    //
    //----------------------------------------------------------------

    sysinline Space size() const // always >= 0
        {return theSize;}

    //----------------------------------------------------------------
    //
    // Get pointer
    //
    //----------------------------------------------------------------

    sysinline Pointer ptrUnsafeForInternalUseOnly() const
        {return thePtr;}

#if HEXLIB_GUARDED_MEMORY

    sysinline typename ArrayPtr(Type) ptr() const
        {return ArrayPtrCreate(Type, thePtr, theSize, DbgptrArrayPreconditions());}

#else

    sysinline Pointer ptr() const
        {return thePtr;}

#endif

    //----------------------------------------------------------------
    //
    // subr
    //
    // Cuts from the array a range of elements given by origin and end: point RIGHT AFTER the last element.
    // If the range does not fit the array, it is clipped to fit the array and false is returned.
    //
    //----------------------------------------------------------------

    template <typename OtherPointer>
    sysinline bool subr(Space org, Space end, ArrayEx<OtherPointer>& result) const
    {
        ARRAY__CHECK_POINTER(Pointer, OtherPointer);

        Space clOrg = clampRange(org, 0, theSize);
        Space clEnd = clampRange(end, clOrg, theSize);

        result.thePtr = &thePtr[clOrg];
        result.theSize = clEnd - clOrg;

        return (clOrg == org) && (clEnd == end);
    }

    //----------------------------------------------------------------
    //
    // subs
    //
    // Cuts from the array a range of elements given by origin and size.
    // If the range does not fit the array, it is clipped to fit the array and false is returned.
    //
    //----------------------------------------------------------------

    template <typename OtherPointer>
    sysinline bool subs(Space org, Space size, ArrayEx<OtherPointer>& result) const
    {
        ARRAY__CHECK_POINTER(Pointer, OtherPointer);

        Space clOrg = clampRange(org, 0, theSize);
        Space clSize = clampRange(size, 0, theSize - clOrg);

        result.thePtr = thePtr + clOrg;
        result.theSize = clSize;

        return (clOrg == org) && (clSize == size);
    }

    //----------------------------------------------------------------
    //
    // exchange
    //
    //----------------------------------------------------------------

    sysinline friend void exchange(ArrayEx<Pointer>& a, ArrayEx<Pointer>& b)
    {
        exchange(a.thePtr, b.thePtr);
        exchange(a.theSize, b.theSize);
    }

    //----------------------------------------------------------------
    //
    // validAccess.
    //
    //----------------------------------------------------------------

    sysinline bool validAccess(Space pos) const
    {
        ARRAY_EXPOSE_EX(*this, my);
        return ARRAY_VALID_ACCESS(my, pos);
    }

    //----------------------------------------------------------------
    //
    // Get pointer, get reference, read element functions:
    // direct access, checked only in guarded mode.
    //
    // As the total size of the array in bytes cannot exceed `spaceMax` bytes,
    // the element address multiplication is performed in the `SpaceU` type.
    //
    // This makes the code more efficient in 64-bit mode, including on GPU.
    //
    //----------------------------------------------------------------

    sysinline auto pointer(Space index) const
        {return addOffset(ptr(), SpaceU(index) * SpaceU(sizeof(Type)));}

    sysinline auto& operator [](Space index) const
        {return pointer(index)[0];}

    sysinline auto read(Space index) const
        {return helpRead(*pointer(index));}

    //----------------------------------------------------------------
    //
    // writeSafe
    //
    //----------------------------------------------------------------

    template <typename Value>
    sysinline void writeSafe(Space index, const Value& value) const
    {
        ARRAY_EXPOSE_UNSAFE_EX(*this, my);

        if (ARRAY_VALID_ACCESS(my, index))
            helpModify(myPtr[index]) = helpRead(value);
    }

};

////

COMPILE_ASSERT_EQUAL_LAYOUT(ArrayBase<int>, ArrayEx<int*>);

//================================================================
//
// hasData 1D
//
//================================================================

template <typename Pointer>
sysinline bool hasData(const ArrayEx<Pointer>& array)
    {return array.size() >= 1;}

sysinline bool hasData(Space size)
    {return size >= 1;}

//================================================================
//
// makeArray
//
//================================================================

template <typename Pointer>
sysinline auto makeArray(Pointer ptr, Space size)
{
    ArrayEx<Pointer> result;
    result.assignUnsafe(ptr, size);
    return result;
}

//================================================================
//
// equalSize support
//
//================================================================

template <>
GET_SIZE_DEFINE(Space, value)

////

template <typename Pointer>
GET_SIZE_DEFINE(ArrayEx<Pointer>, value.size())

////

template <typename Pointer>
sysinline Space getLayers(const ArrayEx<Pointer>& arr)
    {return 1;}

//================================================================
//
// Array<Type>
//
// Array for C++ address space.
//
//================================================================

template <typename Type>
using Array = ArrayEx<Type*>;

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
sysinline const Array<const Type>& makeConst(const Array<Type>& array)
{
    return recastEqualLayout<const Array<const Type>>(array);
}

//================================================================
//
// recastElement
//
// Use with caution!
//
//================================================================

template <typename Dst, typename Src>
sysinline auto& recastElement(const Array<Src>& array)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(Src, Dst);
    return recastEqualLayout<const Array<Dst>>(array);
}
