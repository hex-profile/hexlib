#include "dataAlloc/matrixMemory.inl"
#include "vectorTypes/vectorBase.h"

//================================================================
//
// instantiations
//
//================================================================

#define TMP_MACRO(Type, o) \
    template class MatrixMemoryEx<Type*>;
  
VECTOR_INT_FOREACH(TMP_MACRO, o)
VECTOR_FLOAT_FOREACH(TMP_MACRO, o)

#undef TMP_MACRO

