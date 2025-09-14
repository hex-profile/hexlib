#pragma once

#include <type_traits>

#include "extLib/data/arrayBase.h"
#include "extLib/types/pointTypes.h"

//================================================================
//
// Pitch*
//
//================================================================

struct PitchPositiveOrZero;
struct PitchMayBeNegative;

////

using PitchDefault = PitchPositiveOrZero;

//================================================================
//
// PitchIsEqual
//
//================================================================

template <typename T1, typename T2>
struct PitchIsEqual
{
    static constexpr bool value = false;
};

template <typename T>
struct PitchIsEqual<T, T>
{
    static constexpr bool value = true;
};

//================================================================
//
// PitchCheckConversion
//
//================================================================

template <typename Src, typename Dst>
struct PitchCheckConversion;

template <typename Any>
struct PitchCheckConversion<Any, PitchMayBeNegative> {};

template <>
struct PitchCheckConversion<PitchPositiveOrZero, PitchPositiveOrZero> {};

//================================================================
//
// matrixBaseIsValid
//
// (1) sizeX >= 0 && sizeY >= 0
// (2) sizeX <= |pitch|
// (3) (sizeY * pitch * elemSize) fits into Space type.
//
//================================================================

template <Space elemSize, typename Pitch>
bool matrixBaseIsValid(Space sizeX, Space sizeY, Space pitch);

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

template <typename Type, typename Pointer = Type*, typename Pitch = PitchDefault>
class MatrixBase
{

public:

    HEXLIB_INLINE MatrixBase()
    {
    }

public:

    template <typename OtherPointer>
    HEXLIB_NODISCARD
    HEXLIB_INLINE bool assignValidated(OtherPointer memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        HEXLIB_ENSURE((matrixBaseIsValid<sizeof(Type), Pitch>(sizeX, sizeY, memPitch)));
        assignUnsafe(memPtr, memPitch, sizeX, sizeY);
        return true;
    }

    template <typename OtherPointer>
    HEXLIB_INLINE void assignUnsafe(OtherPointer memPtr, Space memPitch, Space sizeX, Space sizeY)
    {
        constexpr Space maxArea = spaceMax / Space(sizeof(Type));

        if // quick check
        (
            !
            (
                SpaceU(sizeX) <= SpaceU(maxArea) && // mostly to check it's >= 0
                SpaceU(sizeY) <= SpaceU(maxArea) &&
                (PitchIsEqual<Pitch, PitchMayBeNegative>::value || memPitch >= 0)
            )
        )
        {
            sizeX = 0;
            sizeY = 0;
        }

        theMemPtrUnsafe = memPtr;
        theMemPitch = memPitch;
        theSizeX = sizeX;
        theSizeY = sizeY;
    }

public:

    HEXLIB_INLINE Pointer memPtr() const
        {return theMemPtrUnsafe;}

    HEXLIB_INLINE Space memPitch() const
        {return theMemPitch;}

    HEXLIB_INLINE Point<Space> size() const
        {return point(theSizeX, theSizeY);}

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

//================================================================
//
// MatrixBaseAP
//
// An alias for MatrixBase with negative pitch support.
//
//================================================================

template <typename Type, typename Pointer = Type*>
using MatrixBaseAP = MatrixBase<Type, Pointer, PitchMayBeNegative>;
