#pragma once

#include "data/matrix.h"

//================================================================
//
// flipMatrix
//
//================================================================

template <typename Ptr, typename Pitch>
sysinline auto flipMatrix(const MatrixEx<Ptr, Pitch>& img)
{
    MATRIX_EXPOSE_UNSAFE(img);

    ////

    MatrixEx<Ptr, PitchMayBeNegative> result;

    if (imgSizeX >= 1 && imgSizeY >= 1)
    {
        auto flippedPtr = MATRIX_POINTER(img, 0, imgSizeY - 1);
        auto flippedPitch = -imgMemPitch;
        result.assignUnsafe(flippedPtr, flippedPitch, imgSizeX, imgSizeY);
    }

    return result;
}
