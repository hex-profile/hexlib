#pragma once

#include "data/matrix.h"

//================================================================
//
// flipMatrix
//
//================================================================

template <typename Ptr>
inline MatrixEx<Ptr> flipMatrix(const MatrixEx<Ptr>& img)
{
    Ptr imgMemPtr = img.memPtrUnsafeInternalUseOnly();
    Space imgMemPitch = img.memPitch();
    Space imgSizeX = img.sizeX();
    Space imgSizeY = img.sizeY();

    ////

    MatrixEx<Ptr> result;

    if (imgSizeX >= 1 && imgSizeY >= 1)
    {
        Ptr flippedPtr = MATRIX_POINTER(img, 0, imgSizeY - 1);

        Space flippedPitch = -imgMemPitch;
        result.assign(flippedPtr, flippedPitch, imgSizeX, imgSizeY);
    }

    return result;
}
