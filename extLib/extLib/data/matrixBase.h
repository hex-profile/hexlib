#pragma once

#include <type_traits>

#include "extLib/data/arrayBase.h"
#include "extLib/types/pointTypes.h"

//================================================================
//
// MatrixBase
//
//----------------------------------------------------------------
//
// memPtr:
// Points to (0, 0) element. Can be undefined if the matrix is empty.
//
// memPitch:
// The difference of pointers to (X, Y+1) and (X, Y) elements.
// The difference is expressed in elements (not bytes) and can be negative.
//
// sizeX, sizeY:
// The width and height of the matrix. Both are >= 0.
// If either of them is zero, the matrix is empty.
//
//================================================================

template <typename Type, typename Pointer = Type*>
class MatrixBase
{

public:

    HEXLIB_INLINE MatrixBase()
    {
    }

    template <typename OtherPointer>
    HEXLIB_INLINE MatrixBase(OtherPointer memPtr, Space memPitch, Space sizeX, Space sizeY)
        : 
        theMemPtrUnsafe{memPtr},
        theMemPitch{memPitch},
        theSizeX{sizeX},
        theSizeY{sizeY}
    {
    }

protected:

    // Base pointer. If the matrix is empty, can be 0.
    Pointer theMemPtrUnsafe = Pointer(0);

    // Pitch. Can be negative. |pitch| >= sizeX.
    // If the matrix is empty, can be undefined.
    Space theMemPitch = 0;

    // Dimensions. Always >= 0.
    // sizeX >= |pitch|
    Space theSizeX = 0;
    Space theSizeY = 0;

};
