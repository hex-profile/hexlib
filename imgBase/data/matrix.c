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

template <Space elemSize>
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

    Space absPitch = 0;
    check_flag(safeAbs(pitch, absPitch), ok);

    check_flag(sizeX <= absPitch, ok);

    return ok;
}

//----------------------------------------------------------------

template bool matrixParamsAreValid<0x01>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x02>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x04>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x08>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x10>(Space sizeX, Space sizeY, Space pitch);
