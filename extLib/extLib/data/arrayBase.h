#pragma once

#include <type_traits>

#include "extLib/data/space.h"
#include "extLib/types/compileTools.h"

//================================================================
//
// ArrayBase<Type>
//
//----------------------------------------------------------------
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
// ArrayValidityAssertion
//
// size >= 0
// size * sizeof(*ptr) fits into Space type.
//
//================================================================

struct ArrayValidityAssertion
{
};

//================================================================
//
// ArrayBase
//
//================================================================

template <typename Type, typename Pointer = Type*>
class ArrayBase
{

public:

    HEXLIB_INLINE ArrayBase()
    {
    }

    template <typename OtherPointer>
    HEXLIB_INLINE ArrayBase(OtherPointer ptr, Space size)
        :
        thePtr{ptr},
        theSize{size}
    {
    }

protected:

    Pointer thePtr = Pointer(0);
    Space theSize = 0;

};

//================================================================
//
// arrayBaseIsValid
//
//================================================================

template <Space elemSize>
HEXLIB_INLINE bool arrayBaseIsValid(Space size)
{
    HEXLIB_ENSURE(size >= 0);

    ////

    static_assert(elemSize >= 1, "");
    constexpr Space maxArea = spaceMax / elemSize;

    HEXLIB_ENSURE(size <= maxArea);

    return true;
}
