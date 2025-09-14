#include "matrixBase.h"

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
bool matrixBaseIsValid(Space sizeX, Space sizeY, Space pitch)
{
    HEXLIB_ENSURE(sizeX >= 0);
    HEXLIB_ENSURE(sizeY >= 0);

    ////

    static_assert(elemSize >= 1, "");
    constexpr Space maxArea = spaceMax / elemSize;

    ////

    Space absPitch = pitch;

    if (PitchIsEqual<Pitch, PitchMayBeNegative>::value)
    {
        if (absPitch < 0)
            absPitch = -absPitch;
    }

    HEXLIB_ENSURE(absPitch >= 0);

    ////

    HEXLIB_ENSURE(sizeX <= absPitch);

    ////

    if (sizeY >= 1)
    {
        Space maxWidth = maxArea / sizeY;
        HEXLIB_ENSURE(absPitch <= maxWidth);
    }

    return true;
}

//================================================================
//
// matrixBaseIsValid
//
//================================================================

#define TMP_MACRO(n) \
    template bool matrixBaseIsValid<n, PitchMayBeNegative>(Space sizeX, Space sizeY, Space pitch); \
    template bool matrixBaseIsValid<n, PitchPositiveOrZero>(Space sizeX, Space sizeY, Space pitch);

TMP_MACRO(0x01)
TMP_MACRO(0x02)
TMP_MACRO(0x04)
TMP_MACRO(0x08)
TMP_MACRO(0x10)

#undef TMP_MACRO
