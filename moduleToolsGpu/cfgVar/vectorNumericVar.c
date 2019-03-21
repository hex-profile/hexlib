#include "vectorTypes/vectorOperations.h"
#include "cfgVar/vectorSerialization.h"
#include "cfgTools/numericVar.inl"
#include "vectorTypes/vectorType.h"

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class SerializeNumericVar< Type >; \

TMP_MACRO(float32_x2, o)
TMP_MACRO(float32_x4, o)

#undef TMP_MACRO
