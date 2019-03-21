#include "matrixMemory.inl"

#include "numbers/int/intBase.h"
#include "numbers/float/floatBase.h"

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class MatrixMemoryEx<Type*>; \
    template class MatrixMemoryEx<Point<Type>*>;

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO
