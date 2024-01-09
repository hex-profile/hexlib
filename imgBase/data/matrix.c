#include "matrix.inl"

//----------------------------------------------------------------

#define TMP_MACRO(n) \
    template bool matrixParamsAreValid<n, PitchMayBeNegative>(Space sizeX, Space sizeY, Space pitch); \
    template bool matrixParamsAreValid<n, PitchPositiveOrZero>(Space sizeX, Space sizeY, Space pitch);

TMP_MACRO(0x01)
TMP_MACRO(0x02)
TMP_MACRO(0x04)
TMP_MACRO(0x08)
TMP_MACRO(0x10)

#undef TMP_MACRO
