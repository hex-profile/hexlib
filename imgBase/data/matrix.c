#include "matrix.inl"

//----------------------------------------------------------------

template bool matrixParamsAreValid<0x01>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x02>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x04>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x08>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x10>(Space sizeX, Space sizeY, Space pitch);
template bool matrixParamsAreValid<0x20>(Space sizeX, Space sizeY, Space pitch);
