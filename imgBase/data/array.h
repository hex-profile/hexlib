#pragma once

#include "dbgptr/dbgptrGate.h"
#include "data/commonFuncs.h"
#include "data/pointerInterface.h"
#include "numbers/int/intType.h"
#include "extLib/data/arrayBase.h"

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
// USAGE EXAMPLES:
//
//================================================================

#if 0

// Construct empty array.
Array<int> intArray;

// Convert an array to a read-only array.
Array<const int> constArray = intArray;
Array<const int> anotherConstArray = makeConst(intArray);

// Construct array from details: ptr and size.
Array<const uint8> example(srcPtr, srcSize);

// Setup array from details: ptr and size.
example.assign(srcPtr, srcSize);

// Make the array empty:
example.assignNull();

// Access array details (decomposing array is better way):
REQUIRE(example.ptr() != 0);
REQUIRE(example.size() != 0);

// Decompose a array to detail variables:
ARRAY_EXPOSE(example);
REQUIRE(examplePtr != 0);
REQUIRE(exampleSize != 0);

// Example element loop for a decomposed array:
uint32 sum = 0;

for_count (i, exampleSize)
    sum += examplePtr[i];

// Save element range [10, 30) as a new array using
// "subs" (subarray by size) function. Check that no clipping occured.
Array<const uint8> tmp1;
REQUIRE(example.subs(10, 20, tmp1));

// Save element range [10, 30) as a new array using
// "subr" (subarray by rect) function. Check that no clipping occured.
Array<const uint8> tmp2;
REQUIRE(example.subr(10, 30, tmp2));

// Removing const qualifier from elements (avoid this):
Array<uint8> tmp3 = recastElement<uint8>(tmp2);

// Check that arrays have equal size.
REQUIRE(equalSize(example, tmp1, tmp2));
REQUIRE(equalSize(tmp1, tmp2, 20));

// Check that array has non-zero size
REQUIRE(hasData(example));
REQUIRE(hasData(example.size()));

#endif

//================================================================
//
// ARRAY__CHECK_CONVERSION
//
//================================================================

template <typename SrcPointer, typename DstPointer>
inline auto arrayCheckPointerConversion()
{
    SrcPointer srcPtr(0);
    DstPointer dstPtr = srcPtr;
    return dstPtr;
}

//----------------------------------------------------------------

#define ARRAY__CHECK_CONVERSION(SrcPointer, DstPointer) \
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

public:

    using Type = typename PtrElemType<Pointer>::T;

private:

    using BaseType = ArrayBase<Type, Pointer>;

    using BaseType::thePtr;
    using BaseType::theSize;

private:

    template <typename OtherPointer>
    friend class ArrayEx;

    template <typename OtherPointer>
    friend class MatrixEx;

    //
    // Creation
    //

public:

    sysinline ArrayEx()
        {}

    sysinline ArrayEx(Pointer ptr, Space size)
        {assign(ptr, size);}

    //
    // Assign data
    //

public:

    sysinline void assign(Pointer ptr, Space size)
    {
        static const Space maxArraySize = TYPE_MAX(Space) / Space(sizeof(Type));

        bool ok = SpaceU(size) <= SpaceU(maxArraySize); // 0..maxArraySize
        thePtr = ptr;
        theSize = ok ? size : 0;
    }

    sysinline void assign(Pointer ptr, Space size, const ArrayValidityAssertion&)
    {
        thePtr = ptr;
        theSize = size;
    }

    sysinline void assignNull()
    {
        thePtr = Pointer(0);
        theSize = 0;
    }

    //
    // Export cast (no code generated, reinterpret 'this')
    //

public:

    template <typename OtherPointer>
    sysinline operator const ArrayEx<OtherPointer>& () const
    {
        ARRAY__CHECK_CONVERSION(Pointer, OtherPointer);
        return * (const ArrayEx<OtherPointer>*) this;
    }

    //
    // Get size
    //

public:

    sysinline Space size() const // always >= 0
        {return theSize;}

    //
    // Get pointer
    //

    sysinline Pointer ptrUnsafeForInternalUseOnly() const
        {return thePtr;}

#if HEXLIB_GUARDED_MEMORY

    sysinline typename ArrayPtr(Type) ptr() const
        {return ArrayPtrCreate(Type, thePtr, theSize, DbgptrArrayPreconditions());}

#else

    sysinline Pointer ptr() const
        {return thePtr;}

#endif

    //
    // subr
    //
    // Cuts from the array a range of elements given by origin and end: point RIGHT AFTER the last element.
    // If the range does not fit the array, it is clipped to fit the array and false is returned.
    //

    template <typename OtherPointer>
    sysinline bool subr(Space org, Space end, ArrayEx<OtherPointer>& result) const
    {
        ARRAY__CHECK_CONVERSION(Pointer, OtherPointer);

        Space clOrg = clampRange(org, 0, theSize);
        Space clEnd = clampRange(end, clOrg, theSize);

        result.thePtr = &thePtr[clOrg];
        result.theSize = clEnd - clOrg;

        return (clOrg == org) && (clEnd == end);
    }

    //
    // subs
    //
    // Cuts from the array a range of elements given by origin and size.
    // If the range does not fit the array, it is clipped to fit the array and false is returned.
    //

    template <typename OtherPointer>
    sysinline bool subs(Space org, Space size, ArrayEx<OtherPointer>& result) const
    {
        ARRAY__CHECK_CONVERSION(Pointer, OtherPointer);

        Space clOrg = clampRange(org, 0, theSize);
        Space clSize = clampRange(size, 0, theSize - clOrg);

        result.thePtr = thePtr + clOrg;
        result.theSize = clSize;

        return (clOrg == org) && (clSize == size);
    }

    ////

public:

    friend inline void exchange(ArrayEx<Pointer>& a, ArrayEx<Pointer>& b)
    {
        exchange(a.thePtr, b.thePtr);
        exchange(a.theSize, b.theSize);
    }

    //----------------------------------------------------------------
    //
    // validAccess.
    //
    //----------------------------------------------------------------

public:

    sysinline bool validAccess(Space pos) const
    {
        ARRAY_EXPOSE_EX(*this, my);
        return ARRAY_VALID_ACCESS(my, pos);
    }

    //----------------------------------------------------------------
    //
    // Pointer, reference, read: direct access, checked only in guarded mode.
    //
    //----------------------------------------------------------------

    sysinline auto pointer(Space index) const
        {return ptr() + index;}

    sysinline auto& operator [](Space index) const
        {return ptr()[index];}

    sysinline auto read(Space index) const
        {return helpRead(ptr()[index]);}

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
// Array<Type>
//
// Array for C++ address space: identical to ArrayEx<Type*>.
//
//================================================================

template <typename Type>
class Array : public ArrayEx<Type*>
{

public:

    using Base = ArrayEx<Type*>;

    //----------------------------------------------------------------
    //
    // Construct
    //
    //----------------------------------------------------------------

    sysinline Array()
        {}

    sysinline Array(Type* ptr, Space size)
        : Base(ptr, size) {}

    sysinline Array(const Base& base)
        : Base(base) {}

    //----------------------------------------------------------------
    //
    // Export cast (no code generated, reinterpret 'this')
    //
    //----------------------------------------------------------------

    template <typename OtherType>
    sysinline operator const Array<OtherType>& () const
    {
        ARRAY__CHECK_CONVERSION(Type*, OtherType*);
        return recastEqualLayout<const Array<OtherType>>(*this);
    }

    template <typename OtherType>
    sysinline operator const Array<OtherType> () const
    {
        ARRAY__CHECK_CONVERSION(Type*, OtherType*);
        return recastEqualLayout<const Array<OtherType>>(*this);
    }

};

//================================================================
//
// makeArray
//
//================================================================

template <typename Element>
sysinline auto makeArray(Element* ptr, Space size)
    {return Array<Element>(ptr, size);}

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

template <typename Type>
GET_SIZE_DEFINE(Array<Type>, value.size())

////

template <typename Pointer>
sysinline Space getLayers(const ArrayEx<Pointer>& arr)
    {return 1;}

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
sysinline const ArrayEx<const Type*>& makeConst(const ArrayEx<Type*>& array)
{
    return recastEqualLayout<const ArrayEx<const Type*>>(array);
}

//================================================================
//
// recastElement
// 
// Use with caution!
//
//================================================================

template <typename Dst, typename Src>
sysinline const Array<Dst>& recastElement(const Array<Src>& array)
{
    COMPILE_ASSERT_EQUAL_LAYOUT(Src, Dst);
    return recastEqualLayout<const Array<Dst>>(array);
}
