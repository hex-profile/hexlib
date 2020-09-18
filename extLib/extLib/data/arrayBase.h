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
