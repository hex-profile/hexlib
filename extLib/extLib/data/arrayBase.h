#pragma once

#include <type_traits>

#include "extLib/data/space.h"
#include "extLib/types/compileTools.h"

//================================================================
//
// arrayBaseIsValid
//
// size >= 0
// size * sizeof(*ptr) fits into Space type.
//
//================================================================

template <Space elemSize>
HEXLIB_INLINE bool arrayBaseIsValid(Space size)
{
    constexpr Space maxSize = spaceMax / elemSize;
    return SpaceU(size) <= SpaceU(maxSize); // [0..maxSize]
}

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

    template <typename SourcePointer>
    HEXLIB_NODISCARD
    HEXLIB_INLINE bool assignValidated(SourcePointer ptr, Space size)
    {
        HEXLIB_ENSURE(arrayBaseIsValid<sizeof(Type)>(size));

        thePtr = ptr;
        theSize = size;
        return true;
    }

    template <typename SourcePointer>
    HEXLIB_INLINE void assignUnsafe(SourcePointer ptr, Space size)
    {
        if (!arrayBaseIsValid<sizeof(Type)>(size)) // quick check
            size = 0;

        thePtr = ptr;
        theSize = size;
    }

    HEXLIB_INLINE void assignNull()
    {
        thePtr = Pointer(0);
        theSize = 0;
    }

public:

    Pointer data() const
        {return thePtr;}

    Pointer ptr() const
        {return thePtr;}

    Space size() const
        {return theSize;}

protected:

    Pointer thePtr = Pointer(0);
    Space theSize = 0;

};
