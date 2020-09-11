#pragma once

#include <type_traits>

#include "extLib/data/space.h"
#include "extLib/types/compileTools.h"

//================================================================
//
// ArrayBase<Type>
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

template <typename Type>
class ArrayBase
{

    template <typename OtherType>
    friend class MatrixBase;

public:

    HEXLIB_INLINE ArrayBase()
        : theSize(0) {}

    HEXLIB_INLINE ArrayBase(Type* ptr, Space size)
        {assign(ptr, size);}

public:

    HEXLIB_INLINE bool assign(Type* ptr, Space size)
    {
        constexpr Space maxArraySize = spaceMax / Space(sizeof(Type));

        bool ok = SpaceU(size) <= SpaceU(maxArraySize); // [0..maxArraySize]

        thePtr = ptr;
        theSize = ok ? size : 0;

        return ok;
    }

    HEXLIB_INLINE void assignNull()
    {
        thePtr = nullptr;
        theSize = 0;
    }

    //
    // Export cast (no code generated, reinterpret 'this').
    //

public:

    template <typename OtherType>
    HEXLIB_INLINE operator const ArrayBase<OtherType>& () const
    {
        using Src = Type;
        using Dst = OtherType;

        static_assert(std::is_convertible<Src*, Dst*>::value, "");
        static_assert(sizeof(ArrayBase<Src>) == sizeof(ArrayBase<Dst>), "");
        static_assert(alignof(ArrayBase<Src>) == alignof(ArrayBase<Dst>), "");
        return * (const ArrayBase<Dst>*) this;
    }

public:

    //
    // Export const.
    //

    template <typename OtherType>
    HEXLIB_INLINE operator ArrayBase<OtherType> () const
    {
        using Src = Type;
        using Dst = OtherType;

        static_assert(std::is_convertible<Src*, Dst*>::value, "");
        static_assert(sizeof(ArrayBase<Src>) == sizeof(ArrayBase<Dst>), "");
        static_assert(alignof(ArrayBase<Src>) == alignof(ArrayBase<Dst>), "");
        return * (const ArrayBase<Dst>*) this;
    }

    //
    // Access
    //

public:

    HEXLIB_INLINE Space size() const // always >= 0
        {return theSize;}

    HEXLIB_INLINE Type* ptr() const
        {return thePtr;}

    //
    // Data
    //

private:

    Type* thePtr; // if theSize == 0, is not used.
    Space theSize; // always >= 0

};

//================================================================
//
// hasData
//
//================================================================

template <typename Type>
HEXLIB_INLINE bool hasData(const ArrayBase<Type>& array)
{
    return array.size() >= 1;
}

//================================================================
//
// makeConst (fast)
//
//================================================================

template <typename Type>
HEXLIB_INLINE const ArrayBase<const Type>& makeConst(const ArrayBase<Type>& array)
{
    static_assert(sizeof(ArrayBase<const Type>) == sizeof(ArrayBase<Type>), "");
    return * (const ArrayBase<const Type>*) &array;
}
