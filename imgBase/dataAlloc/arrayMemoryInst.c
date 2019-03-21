#include "arrayMemory.inl"

#include "numbers/int/intBase.h"
#include "numbers/float/floatBase.h"
#include "charType/charType.h"

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class ArrayMemoryEx<Type*>; \
    template class ArrayMemoryEx<Point<Type>*>;

BUILTIN_INT_FOREACH(TMP_MACRO, o)
BUILTIN_FLOAT_FOREACH(TMP_MACRO, o)

TMP_MACRO(CharType, o)

#undef TMP_MACRO
