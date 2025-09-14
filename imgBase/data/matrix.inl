#include "matrix.h"

#include "data/spacex.h"

//================================================================
//
// matrixParamsAreValid
//
// (1) sizeX >= 0 && sizeY >= 0
// (2) sizeX <= |pitch|
// (3) (sizeY * pitch * elemSize) fits into Space type.
//
// ~30 instructions
//
//================================================================

template <Space elemSize, typename Pitch>
bool matrixParamsAreValid(Space sizeX, Space sizeY, Space pitch)
{
    COMPILE_ASSERT(elemSize >= 1);

    ////

    bool ok = true;

    check_flag(sizeX >= 0, ok);
    check_flag(sizeY >= 0, ok);

    ////

    Space pitchByHeight = 0;
    check_flag(safeMul(sizeY, pitch, pitchByHeight), ok);

    Space pitchByHeightBytes = 0;
    check_flag(safeMul(pitchByHeight, elemSize, pitchByHeightBytes), ok);

    ////

    Space absPitch = pitch;

    if (TYPE_EQUAL(Pitch, PitchPositiveOrZero))
        check_flag(pitch >= 0, ok);
    else
        check_flag(safeAbs(pitch, absPitch), ok);

    ////

    check_flag(sizeX <= absPitch, ok);

    return ok;
}
